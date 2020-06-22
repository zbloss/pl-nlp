import logging
import pytorch_lightning as pl
from src.models.minst_model import MNISTModel


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger = logging.getLogger(__name__)
    logger.info('Begining Training')

    model = MNISTModel()
    logger.info('Built model')

    trainer = pl.Trainer(max_epochs=20)
    
    logger.info('Beginning Training')
    trainer.fit(model)
    logger.info('Training complete')

