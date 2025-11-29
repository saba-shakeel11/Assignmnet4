# MLOps Kubeflow Assignment

## Project Overview
This project implements an MLOps pipeline for predicting housing prices using the Boston Housing dataset. It uses DVC for data versioning, Kubeflow Pipelines for orchestration, and GitHub Actions/Jenkins for CI.

The ML problem is regression: predicting median house values based on features like crime rate, rooms, etc.

## Setup Instructions
1. **Minikube and Kubeflow Pipelines**:
   - Install Minikube: `curl -LO https://storage.googleapis.com/minikube/releases/latest/minikube-windows-amd64.exe` (for Windows).
   - Start: `minikube start --driver=hyperv` (or virtualbox).
   - Deploy KFP: Follow https://www.kubeflow.org/docs/components/pipelines/installation/standalone-deployment/.
   - Access dashboard: `minikube service -n kubeflow ml-pipeline-ui`.

2. **DVC Remote Storage**:
   - Initialize: `dvc init`.
   - Add remote: `dvc remote add myremote path/to/local/folder` (or use Google Drive: install dvc[gs], `dvc remote add myremote gdrive://folder_id`).
   - Track data: Download Boston CSV from https://www.kaggle.com/datasets/altavish/boston-housing-dataset, rename to raw_data.csv, place in data/, then `dvc add data/raw_data.csv`, `dvc push`.

## Pipeline Walkthrough
1. Install dependencies: `pip install -r requirements.txt`.
2. Compile components: Run `python src/pipeline_components.py` to generate YAML in components/.
3. Compile pipeline: `python pipeline.py` to generate pipeline.yaml.
4. Upload to KFP UI: Create experiment, upload pipeline.yaml, run it.
5. Monitor in KFP dashboard.