

import importlib
import argparse
import pytorch_lightning as pl
from covid_detector.models.util import create_model
from covid_detector import lit_models


def _setup_parser():

    # Get a parser with pl.Trainer args
    parser = argparse.ArgumentParser(add_help=False)
    trainer_parser = pl.Trainer.add_argparse_args(parser)
    trainer_parser._action_groups[1].title = 'Trainer args'

    # Initialise another parser 
    parser = argparse.ArgumentParser(add_help=False, parents=[trainer_parser])

    # Set some default args
    parser.add_argument('--data_class', type=str, default='Abnormality')
    parser.add_argument('--model_class', type=str, default='efficientdet_d0')
    parser.add_argument('--load_from_checkpoint', type=str, default=None)

    # Import default data_class
    temp_args, _ = parser.parse_known_args()
    data_class = _import_class(f'covid_detector.data.{temp_args.data_class}')

    # Add data, lit model args to parser
    data_group = parser.add_argument_group('Data Args')
    data_class.add_to_argparse(data_group)

    lit_model_group = parser.add_argument_group('Lit Model Args')
    lit_models.BaseLitModel.add_to_argparse(lit_model_group)

    parser.add_argument('--help', '-h', action='help')

    return parser



def _import_class(module_class_string: str = None):
    module_str, class_str = module_class_string.rsplit('.', 1)
    module = importlib.import_module(module_str)
    return getattr(module, class_str)


def main():
    '''
    Run an experiment.
    '''
    parser = _setup_parser()
    args = parser.parse_args()
    print('Parsed args')
    data_class = _import_class(f'covid_detector.data.{args.data_class}')

    data = data_class(args)
    model = create_model(name=args.model_class, data=data)

    lit_model_class = lit_models.BaseLitModel
    print('Create Lightning module')
    if args.load_from_checkpoint is not None:
        lit_model = lit_model_class.load_from_checkpoint(
            args.load_from_checkpoint, args=args, model=model)
    else:
        lit_model = lit_model_class(args=args, model=model)

    logger = pl.loggers.TensorBoardLogger(save_dir='training/logs')

    early_stopping_callback = pl.callbacks.EarlyStopping(
        monitor='val_loss', mode='min', patience=10)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        filename='{epoch:03d}-{val_loss:.3f}', monitor='val_loss', mode='min')

    callbacks = [early_stopping_callback, model_checkpoint_callback]

    args.weight_summary = 'full'
    trainer = pl.Trainer.from_argparse_args(
        args, 
        callbacks=callbacks, logger=logger, 
        weights_save_path='training/logs')
    print('Created pl.Trainer')

    trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


    


if __name__ == '__main__':
    main()
