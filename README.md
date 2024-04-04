# Object Detection

Temporary repo created as a proof of concept for a Object Detection application.

## Setup

Before starting this procedure, make sure you have [Python 3.10](https://www.python.org/downloads/) installed on your system. 

## Step 1: Clone the Repository

Clone this repository to your local machine using Git:

```bash
git clone https://github.com/elkike8/object-detection.git
cd object-detection
```

## Step 2: Activate Python Venv

Initialize the python environment by running the following command inside the project folder.

```bash
python -m venv venv
```

Activate the venv by running the following on **Windows**:

```bash
venv\Scripts\activate
```

or on **Mac** / **Linux**:
```bash
source venv/bin/activate
```


## Step 3: Install requirements

All required python packages are found in ```requirements.txt``` and can be installed by running:
```bash
pip install -r requirements.txt
```
## Step 4: Add the videos to be tested

There is a directory in place called `/test-videos/`. Save any video you want the app to have access to there. This directory can be modified in the `app.py` file if desired.

## Step 5: Run the app

After creating the environment and installing the requirements use the run the app using the command:

```
streamlit run app.py
```