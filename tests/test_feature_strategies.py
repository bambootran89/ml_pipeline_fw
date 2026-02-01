"""
Unit tests for feature retrieval strategies.

Tests cover strategy creation, timeseries retrieval,
tabular retrieval, and facade integration.
"""

from unittest.mock import Mock, patch

import pandas as pd
import pytest
from omegaconf import OmegaConf

from mlproject.src.features.facade import FeatureStoreFacade
from mlproject.src.features.strategies import (
    OnlineRetrievalStrategy,
    StrategyFactory,
    TabularRetrievalStrategy,
    TimeseriesRetrievalStrategy,
)


class TestStrategyFactory:
    """Test suite for StrategyFactory."""

    def test_create_timeseries_strategy(self):
        """Factory creates timeseries strategy correctly."""
        strategy = StrategyFactory.create("timeseries")
        assert isinstance(strategy, TimeseriesRetrievalStrategy)

    def test_create_tabular_strategy(self):
        """Factory creates tabular strategy correctly."""
        strategy = StrategyFactory.create("tabular")
        assert isinstance(strategy, TabularRetrievalStrategy)

    def test_create_invalid_type_raises(self):
        """Factory raises on unknown data type."""
        with pytest.raises(ValueError, match="Invalid data_type"):
            StrategyFactory.create("invalid_type")

    def test_create_online_strategy_with_mode(self):
        """Factory creates online strategy when mode=online."""
        strategy = StrategyFactory.create(
            data_type="timeseries",
            mode="online",
            time_point="now",
        )
        assert isinstance(strategy, OnlineRetrievalStrategy)
        assert strategy.time_point == "now"

    def test_create_historical_by_default(self):
        """Factory creates historical strategy by default."""
        strategy = StrategyFactory.create(data_type="timeseries")
        assert isinstance(strategy, TimeseriesRetrievalStrategy)
        assert not isinstance(strategy, OnlineRetrievalStrategy)


class TestTimeseriesStrategy:
    """Test suite for timeseries retrieval strategy."""

    @patch("mlproject.src.features.strategies.TimeSeriesFeatureStore")
    def test_retrieve_calls_store_correctly(self, mock_ts_store_class):
        """Strategy delegates to TimeSeriesFeatureStore properly."""
        mock_store = Mock()
        mock_ts_store = Mock()
        mock_ts_store_class.return_value = mock_ts_store

        expected_df = pd.DataFrame({"value": [1, 2, 3]})
        mock_ts_store.get_sequence_by_range.return_value = expected_df

        strategy = TimeseriesRetrievalStrategy()
        config = {
            "start_date": "2024-01-01T00:00:00+00:00",
            "end_date": "2024-01-02T00:00:00+00:00",
        }

        result = strategy.retrieve(
            store=mock_store,
            features=["view:feat1"],
            entity_key="location_id",
            entity_ids=[1],
            config=config,
        )

        assert result.equals(expected_df)
        mock_ts_store.get_sequence_by_range.assert_called_once()


class TestFeatureStoreFacade:
    """Test suite for FeatureStoreFacade."""

    def test_facade_validates_missing_config(self):
        """Facade raises when required config keys are missing."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "feast://repo",
                    "type": "timeseries",
                    # Missing: featureview, features
                }
            }
        )

        with pytest.raises(ValueError, match="Missing config keys"):
            facade = FeatureStoreFacade(cfg)

    @patch("mlproject.src.features.facade.FeatureStoreFactory")
    @patch("mlproject.src.features.facade.StrategyFactory")
    def test_facade_loads_timeseries(self, mock_strategy_factory, mock_store_factory):
        """Facade successfully loads timeseries features."""
        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "feast://test_repo",
                    "type": "timeseries",
                    "featureview": "test_view",
                    "features": ["feat1", "feat2"],
                    "entity_key": "id",
                    "entity_ids": [1],
                    "start_date": "2024-01-01T00:00:00+00:00",
                    "end_date": "2024-01-02T00:00:00+00:00",
                    "index_col": "timestamp",
                }
            }
        )

        mock_store = Mock()
        mock_store_factory.create.return_value = mock_store

        mock_strategy = Mock()
        expected_df = pd.DataFrame(
            {
                "timestamp": pd.date_range("2024-01-01", periods=3, freq="H"),
                "feat1": [1, 2, 3],
                "feat2": [4, 5, 6],
            }
        )
        mock_strategy.retrieve.return_value = expected_df
        mock_strategy_factory.create.return_value = mock_strategy

        facade = FeatureStoreFacade(cfg)
        result = facade.load_features()

        assert "timestamp" in result.index.name or len(result) > 0
        mock_store_factory.create.assert_called_once_with(
            store_type="feast",
            repo_path="test_repo",
        )
        # Verify factory called WITHOUT store parameter
        mock_strategy_factory.create.assert_called_once()
        call_kwargs = mock_strategy_factory.create.call_args[1]
        assert "store" not in call_kwargs
        assert call_kwargs["data_type"] == "timeseries"


class TestOnlineRetrievalStrategy:
    """Test online retrieval strategy for serving."""

    def test_tabular_retrieves_from_online_store(self):
        """Strategy retrieves single point from Online Store."""
        mock_store = Mock()
        mock_store.get_online_features.return_value = [{"temp": 25.5, "humidity": 60.0}]

        # Factory KHÔNG nhận store
        strategy = OnlineRetrievalStrategy(time_point="now")

        features = ["hourly:temp", "hourly:humidity"]
        config = {
            "data_type": "tabular",
            "featureview": "hourly",
            "features": ["temp", "humidity"],
        }

        # Pass store vào retrieve()
        df = strategy.retrieve(
            store=mock_store,
            features=features,
            entity_key="location_id",
            entity_ids=[42],
            config=config,
        )

        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1
        mock_store.get_online_features.assert_called_once_with(
            entity_rows=[{"location_id": 42}],
            features=features,
        )

    def test_tabular_raises_when_no_online_data(self):
        """Strategy raises ValueError when Online Store returns empty."""
        mock_store = Mock()
        mock_store.get_online_features.return_value = []

        strategy = OnlineRetrievalStrategy(time_point="now")

        features = ["hourly:temp"]
        config = {
            "data_type": "tabular",
            "featureview": "hourly",
            "features": ["temp"],
        }

        with pytest.raises(ValueError, match="No online data found"):
            strategy.retrieve(
                store=mock_store,
                features=features,
                entity_key="location_id",
                entity_ids=[999],
                config=config,
            )

    @patch("mlproject.src.features.strategies.TimeSeriesFeatureStore")
    def test_timeseries_raises_when_no_data(self, mock_ts_store_class):
        """Strategy raises when no data at time_point or end_date."""
        mock_ts_store = Mock()
        mock_ts_store.get_latest_n_sequence_single.return_value = pd.DataFrame()
        mock_ts_store_class.return_value = mock_ts_store

        mock_store = Mock()
        strategy = OnlineRetrievalStrategy(time_point="now")

        features = ["hourly:temp"]
        config = {
            "data_type": "timeseries",
            "featureview": "hourly",
            "features": ["temp"],
            "input_chunk_length": 10,
            "frequency_hours": 1,
            "index_col": "event_timestamp",
        }

        with pytest.raises(ValueError, match="No timeseries data at time_point"):
            strategy.retrieve(
                store=mock_store,
                features=features,
                entity_key="location_id",
                entity_ids=[42],
                config=config,
            )

    def test_missing_features_raises(self):
        """Strategy raises when features are missing."""
        mock_store = Mock()
        strategy = OnlineRetrievalStrategy(time_point="now")

        config = {
            "data_type": "tabular",
            "featureview": "hourly",
            # Missing: features
        }

        with pytest.raises(ValueError, match="must contain 'features'"):
            strategy.retrieve(
                store=mock_store,
                features=["hourly:temp"],
                entity_key="location_id",
                entity_ids=[42],
                config=config,
            )

    def test_missing_featureview_raises(self):
        """Strategy raises when featureview is missing."""
        mock_store = Mock()
        strategy = OnlineRetrievalStrategy(time_point="now")

        config = {
            "data_type": "tabular",
            "features": ["temp"],
            # Missing: featureview
        }

        with pytest.raises(ValueError, match="featureview"):
            strategy.retrieve(
                store=mock_store,
                features=["view:temp"],
                entity_key="location_id",
                entity_ids=[42],
                config=config,
            )


class TestStrategyFactoryWithOnlineMode:
    """Test StrategyFactory with online mode parameter."""

    def test_creates_online_strategy_with_time_point(self):
        """Factory creates OnlineStrategy when mode=online."""
        # Factory KHÔNG nhận store!
        strategy = StrategyFactory.create(
            data_type="timeseries",
            mode="online",
            time_point="2024-01-01T12:00:00",
        )

        assert isinstance(strategy, OnlineRetrievalStrategy)
        assert strategy.time_point == "2024-01-01T12:00:00"

    def test_creates_historical_strategy_by_default(self):
        """Factory creates historical strategy when mode not specified."""
        strategy = StrategyFactory.create(data_type="timeseries")

        # Should NOT be OnlineRetrievalStrategy
        assert isinstance(strategy, TimeseriesRetrievalStrategy)
        assert not isinstance(strategy, OnlineRetrievalStrategy)

    def test_creates_tabular_strategy(self):
        """Factory creates TabularStrategy for tabular data."""
        strategy = StrategyFactory.create(data_type="tabular")

        assert isinstance(strategy, TabularRetrievalStrategy)

    def test_online_strategy_has_no_store_attribute(self):
        """OnlineStrategy does not store reference to store."""
        strategy = StrategyFactory.create(
            data_type="timeseries",
            mode="online",
            time_point="now",
        )

        assert isinstance(strategy, OnlineRetrievalStrategy)
        assert not hasattr(strategy, "store")
        assert hasattr(strategy, "time_point")


class TestFeatureFacadeWithOnlineMode:
    """Test FeatureStoreFacade with online mode parameter."""

    @patch("mlproject.src.features.facade.FeatureStoreFactory")
    @patch("mlproject.src.features.facade.StrategyFactory")
    def test_facade_creates_online_strategy(
        self, mock_strategy_factory, mock_store_factory
    ):
        """Facade creates OnlineStrategy when mode=online."""
        mock_store = Mock()
        mock_store_factory.create.return_value = mock_store

        mock_strategy = Mock()
        mock_strategy.retrieve.return_value = pd.DataFrame(
            {
                "temp": [25.5],
            }
        )
        mock_strategy_factory.create.return_value = mock_strategy

        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "feast://weather",
                    "type": "timeseries",
                    "featureview": "hourly",
                    "features": ["temp"],
                    "entity_key": "location_id",
                    "entity_ids": [42],
                },
                "experiment": {"hyperparams": {"input_chunk_length": 24}},
            }
        )

        facade = FeatureStoreFacade(cfg, mode="online")
        df = facade.load_features(time_point="2024-01-01T12:00:00")

        # Verify StrategyFactory was called correctly
        mock_strategy_factory.create.assert_called_once()
        call_kwargs = mock_strategy_factory.create.call_args[1]

        # CRITICAL: Verify store NOT passed to factory
        assert "store" not in call_kwargs

        # Verify correct parameters
        assert call_kwargs["mode"] == "online"
        assert call_kwargs["time_point"] == "2024-01-01T12:00:00"
        assert call_kwargs["data_type"] == "timeseries"

        # Verify strategy.retrieve was called with store
        mock_strategy.retrieve.assert_called_once()
        retrieve_kwargs = mock_strategy.retrieve.call_args[1]
        assert retrieve_kwargs["store"] is mock_store

        assert isinstance(df, pd.DataFrame)

    @patch("mlproject.src.features.facade.FeatureStoreFactory")
    @patch("mlproject.src.features.facade.StrategyFactory")
    def test_facade_defaults_to_historical_mode(
        self, mock_strategy_factory, mock_store_factory
    ):
        """Facade uses historical mode by default."""
        mock_store = Mock()
        mock_store_factory.create.return_value = mock_store

        mock_strategy = Mock()
        mock_strategy.retrieve.return_value = pd.DataFrame(
            {
                "temp": [25.5],
            }
        )
        mock_strategy_factory.create.return_value = mock_strategy

        cfg = OmegaConf.create(
            {
                "data": {
                    "path": "feast://weather",
                    "type": "timeseries",
                    "featureview": "hourly",
                    "features": ["temp"],
                    "start_date": "2024-01-01",
                    "end_date": "2024-01-02",
                }
            }
        )

        # Don't specify mode
        facade = FeatureStoreFacade(cfg)
        df = facade.load_features()

        # Verify mode defaults to historical
        call_kwargs = mock_strategy_factory.create.call_args[1]
        assert call_kwargs["mode"] == "historical"

        assert isinstance(df, pd.DataFrame)
