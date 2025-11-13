
# Data Flow & Architecture

[Sensors & Cameras] --> [Edge Node (ESP32 / Microcontroller)]
    - sample sensors every N minutes
    - perform lightweight checks (threshold alerts)
    - buffer & batch data

--> [Gateway (Raspberry Pi)]
    - preprocess & aggregate
    - run local inference for immediate control (irrigation)
    - store locally and upload daily summaries

--> [Cloud / Central Server]
    - historical storage, heavy model training (yield prediction)
    - dashboards and remote analytics
    - federated learning orchestration (optional)

--> [Farmer Dashboard / Mobile App]
    - visualizations, manual overrides, recommendations
