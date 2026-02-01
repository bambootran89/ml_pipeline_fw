"""Helper script to generate realistic API test data."""

import json
from datetime import datetime, timedelta


def generate_test_data(num_timesteps=24):
    """Generate realistic test data matching ETTh1 dataset structure.

    Args:
        num_timesteps: Number of timesteps (default 24 for input_chunk_length)

    Returns:
        Dictionary with date, HUFL, MUFL, mobility_inflow
    """
    # Start datetime
    start_date = datetime(2020, 1, 1, 0, 0, 0)

    # Generate dates (hourly)
    dates = [
        (start_date + timedelta(hours=i)).strftime("%Y-%m-%d %H:%M:%S")
        for i in range(num_timesteps)
    ]

    # Generate HUFL values (High UseFul Load - typical range 5-10)
    hufl = [
        5.827,
        5.8,
        5.969,
        6.372,
        7.153,
        7.976,
        8.715,
        9.340,
        9.763,
        9.986,
        10.040,
        9.916,
        9.609,
        9.156,
        8.591,
        7.970,
        7.338,
        6.745,
        6.233,
        5.838,
        5.582,
        5.465,
        5.465,
        5.557,
    ]

    # Generate MUFL values (Middle UseFul Load - typical range 1-3)
    mufl = [
        1.599,
        1.492,
        1.492,
        1.492,
        1.492,
        1.509,
        1.582,
        1.711,
        1.896,
        2.113,
        2.337,
        2.552,
        2.742,
        2.902,
        3.024,
        3.104,
        3.137,
        3.125,
        3.067,
        2.969,
        2.838,
        2.683,
        2.515,
        2.346,
    ]

    # Generate mobility_inflow (typical range 1-5)
    mobility_inflow = [
        1.234,
        1.456,
        1.678,
        1.890,
        2.123,
        2.456,
        2.789,
        3.012,
        3.234,
        3.456,
        3.678,
        3.890,
        4.012,
        4.123,
        4.234,
        4.345,
        4.456,
        4.567,
        4.678,
        4.789,
        4.890,
        4.901,
        4.912,
        4.923,
    ]

    return {
        "date": dates[:num_timesteps],
        "HUFL": hufl[:num_timesteps],
        "MUFL": mufl[:num_timesteps],
        "mobility_inflow": mobility_inflow[:num_timesteps],
    }


def generate_curl_example():
    """Generate curl command with realistic data."""
    data = generate_test_data(24)
    payload = {"data": data}

    return f"""curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{json.dumps(payload, indent=2)}'"""


def generate_python_example():
    """Generate Python requests example."""
    data = generate_test_data(24)

    code = """import requests

# Health check
response = requests.get("http://localhost:8000/health")
print(response.json())

# Prediction with realistic ETTh1 dataset structure
payload = {
    "data": {
"""

    for key in ["date", "HUFL", "MUFL", "mobility_inflow"]:
        code += f'        "{key}": [\n'
        values = data[key]
        # Split into chunks for readability
        for i in range(0, len(values), 6):
            chunk = values[i : i + 6]
            if key == "date":
                chunk_str = ", ".join([f'"{v}"' for v in chunk])
            else:
                chunk_str = ", ".join([str(v) for v in chunk])
            code += f"            {chunk_str},\n"
        code = code.rstrip(",\n") + "\n"
        code += "        ],\n"

    code += """    }
}

response = requests.post(
    "http://localhost:8000/predict",
    json=payload
)
print(response.json())
"""

    return code


if __name__ == "__main__":
    print("Curl example:")
    print(generate_curl_example())
    print("\n" + "=" * 60 + "\n")
    print("Python example:")
    print(generate_python_example())
