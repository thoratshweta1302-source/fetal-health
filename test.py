import requests

sample_input = {
    "baseline value": 120.0,
    "accelerations": 0.0,
    "fetal_movement": 0.0,
    "uterine_contractions": 0.0,
    "light_decelerations": 0.0,
    "severe_decelerations": 0.0,
    "prolongued_decelerations": 0.0,
    "abnormal_short_term_variability": 73.0,
    "mean_value_of_short_term_variability": 0.5,
    "percentage_of_time_with_abnormal_long_term_variability": 43.0,
    "mean_value_of_long_term_variability": 2.4,
    "histogram_width": 64.0,
    "histogram_min": 62.0,
    "histogram_max": 126.0,
    "histogram_number_of_peaks": 2.0,
    "histogram_number_of_zeroes": 0.0,
    "histogram_mode": 120.0,
    "histogram_mean": 137.0,
    "histogram_median": 121.0,
    "histogram_variance": 73.0,
    "histogram_tendency": 1.0
}

response = requests.post('http://127.0.0.1:5000/predict', json=sample_input)
print(response.json())