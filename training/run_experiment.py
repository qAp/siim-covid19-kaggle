
import pytorch_lightning as pl
from covid_detector.models.util import create_model

def _setup_parser():
    pass

def _import_class():
    pass

def main():
    '''
    Run an experiment.
    '''
    parser = _setup_parser()
    args = parser.parse_args()

    data_class = _import_class(f'covid_detector.data.{args.data_class}')
    model_class = args.model_class

    data = data_class(args)
    model = create_model(name=args.model_name, data=data)

    lit_model_class = lit_models.BaseLitModel

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

    trainer.tune(lit_model, datamodule=data)
    trainer.fit(lit_model, datamodule=data)
    trainer.test(lit_model, datamodule=data)


    


if __name__ == 'main':
    main()
