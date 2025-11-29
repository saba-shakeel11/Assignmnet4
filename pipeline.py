import kfp.dsl as dsl
import kfp.compiler as compiler
from src.pipeline_components import data_extraction, data_preprocessing, model_training, model_evaluation

@dsl.pipeline(
    name='MLOps Boston Housing Pipeline',
    description='End-to-end ML pipeline for Boston Housing regression'
)
def ml_pipeline():
    extract = data_extraction()
    preprocess = data_preprocessing(input_path=extract.output)
    train = model_training(
        train_x_path=preprocess.outputs['train_x_path'],
        train_y_path=preprocess.outputs['train_y_path']
    )
    evaluate = model_evaluation(
        test_x_path=preprocess.outputs['test_x_path'],
        test_y_path=preprocess.outputs['test_y_path'],
        model_path=train.output
    )

if __name__ == '__main__':
    compiler.Compiler().compile(ml_pipeline, 'pipeline.yaml')