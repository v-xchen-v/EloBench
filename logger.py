import logging

# Step 1 & 2: Create a logger
info_logger = logging.getLogger('info_logger')
# Step 3: Set log level
info_logger.setLevel(logging.INFO)

# Step 4: Create a handler (e.g., console handler)
info_console_handler = logging.StreamHandler()
info_console_handler.setLevel(logging.INFO)

issue_question_file_handler = logging.FileHandler(r'logs/issue_question.log')
issue_question_file_handler.setLevel(logging.INFO)

# Step 5: Create a formatter and set it to the handler
info_formatter = logging.Formatter('[%(levelname)s] - %(message)s')
info_console_handler.setFormatter(info_formatter)

# Step 6: Add the handler to the logger
info_logger.addHandler(info_console_handler)


# Set up the logger
logger = logging.getLogger('elo_bench_logger')
logger.setLevel(logging.DEBUG)

gpt_as_judger_logger = logging.getLogger('gpt_as_judger_logger')
gpt_as_judger_logger.setLevel(logging.DEBUG)

iterate_to_no_tie_logger = logging.getLogger('iterate_to_n_no_tie')
iterate_to_no_tie_logger.setLevel(logging.DEBUG)

battle_pipeline_logger = logging.getLogger('battle_pipeline_logger')
iterate_to_no_tie_logger.setLevel(logging.INFO)

elo_rating_history_logger = logging.getLogger('elo_rating_history_logger')
elo_rating_history_logger.setLevel(logging.DEBUG)

# Create a console handler and set its level
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Create a logger to record issue questions in questions et
issue_question_logger = logging.getLogger('issue_question_logger')
issue_question_logger.setLevel(logging.DEBUG)
issue_question_logger.addHandler(console_handler)
issue_question_logger.addHandler(issue_question_file_handler)

# # Create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
short_formater = logging.Formatter('[%(levelname)s] %(message)s')

# # Add formatter to console_handler
console_handler.setFormatter(short_formater)
info_console_handler.setFormatter(short_formater)
issue_question_file_handler.setFormatter(short_formater)
# Add console_handler to logger
logger.addHandler(console_handler)
iterate_to_no_tie_logger.addHandler(console_handler)
battle_pipeline_logger.addHandler(info_console_handler)
elo_rating_history_logger.addHandler(console_handler)
gpt_as_judger_logger.addHandler(console_handler)