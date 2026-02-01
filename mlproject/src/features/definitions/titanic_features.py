from typing import Any

# Schema matching Titanic dataset (assumes categorical columns are encoded)
TITANIC_SCHEMA = {
    "Pclass": "int",
    "Age": "float",
    "SibSp": "int",
    "Parch": "int",
    "Fare": "float",
    "Sex": "string",  # Label encoded
    "Embarked": "string",  # Label encoded
    "Survived": "int",
}


def register_titanic_features(store: Any, entity_col: str, source_path: str) -> None:
    """
    Register Titanic dataset features to a Feature Store.

    This function performs the following operations:
    1. Registers the entity representing a Titanic passenger.
    2. Registers a Feature View named 'titanic_view' that includes all
       columns defined in TITANIC_SCHEMA.
    3. Sets the TTL to None since the dataset is static/tabular and does not
       require automatic expiration.

    Args:
        store (Any): An instance of a Feature Store created from a Factory.
        entity_col (str): The column name to be used as the entity key
                          (e.g., passenger ID).
        source_path (str): File path to the source data (CSV or Parquet) that
                           contains the feature values.

    Raises:
        ValueError: If the store cannot register the entity or feature view.
    """
    # Register the passenger entity
    store.register_entity(
        name="passenger",
        join_key=entity_col,
        description="Titanic passenger ID",
        value_type="int",
    )

    # Register the Titanic feature view
    store.register_feature_view(
        name="titanic_view",
        entities=["passenger"],
        schema=TITANIC_SCHEMA,
        source_path=source_path,
        ttl_days=None,
    )
