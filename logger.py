import logging

# Set up the logger
logger = logging.getLogger('elo_bench_logger')
logger.setLevel(logging.DEBUG)

gpt_as_judger_logger = logging.getLogger('gpt_as_judger_logger')
gpt_as_judger_logger.setLevel(logging.DEBUG)

iterate_to_no_tie_logger = logging.getLogger('iterate_to_n_no_tie')
iterate_to_no_tie_logger.setLevel(logging.DEBUG)

battle_pipeline_logger = logging.getLogger('battle_pipeline_logger')
iterate_to_no_tie_logger.setLevel(logging.INFO)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# # Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# # Add formatter to console_handler
console_handler.setFormatter(formatter)

# Add console_handler to logger
logger.addHandler(console_handler)
iterate_to_no_tie_logger.addHandler(console_handler)
battle_pipeline_logger.addHandler(console_handler)