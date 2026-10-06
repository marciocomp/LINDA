# @author: Marcio Lopes

import logging

logging.basicConfig(
        format='%(asctime)s - '
               '%(levelname)-8s '
               '[%(filename)s/'
               '%(module)s/'
               '%(funcName)s/ '
               'line[%(lineno)d]:'
               '\t%(message)s',
        datefmt='(%Y-%m-%d) %H:%M:%S',
        level=logging.INFO
        )

logger = logging.getLogger(__name__)