import sys, os
sys.path.append(os.path.dirname(os.path.dirname((os.path.dirname(__file__)))))

from battle_pipe import BattlePipeline
from datamodel.question_and_answers_collection import QuestionAndAnswersCollection, LLMAnswer
from datamodel import QuestionCollection, PairwiseBattleArrangement
import pandas as pd
from datamodel import MODEL_A_COLUMN_NAME, MODEL_B_COLUMN_NAME
from pathlib import Path

question_file = Path('data')/'arena_data'/'chatbot_arena_conversations'/'chatbot_arena_questions.csv'
q_and_as_file = Path('data')/'arena_data'/'chatbot_arena_conversations'/'chatbot_arena_q_and_as.csv'
battle_arrangement_file = Path('data')/'arena_data'/'chatbot_arena_conversations'/'chatbot_arena_battle_arrangement.csv'
battled_pairs_file = Path('data')/'arena_data/chatbot_arena_conversations'/'chatbot_arena_battled_pairs.csv'

class BattleArenaWithGPT4AsJudger(BattlePipeline):
    def gen_model_answers(self) -> None:
        battle_arrangement = pd.read_csv(battle_arrangement_file)
        question_and_answers = pd.read_csv(q_and_as_file)
        
        def pick_answer(question: str, model_a_or_b: str):
            model_name = battle_arrangement[battle_arrangement['question'] == question].iloc[0][model_a_or_b]

            answer = question_and_answers[question_and_answers['question']==question].iloc[0][model_name]
            return answer

        for rnd in self.battle_arrangements.battles_in_order:
            q = rnd.question
            m_a = rnd.model_a
            m_b = rnd.model_b
            
            # generate ans
            ans_a = pick_answer(q, MODEL_A_COLUMN_NAME)
            ans_b = pick_answer(q, MODEL_B_COLUMN_NAME)

            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_a, ans_a))
            self.question_and_answers_collection.add_answer(q, LLMAnswer(m_b, ans_b))
            
if __name__ == '__main__':           
    bp = BattleArenaWithGPT4AsJudger(save_path='log_battle_arena_gpt4_as_judger_test.csv', num_of_max_battle=1)
    
    # qs = ["Give me a question about animal.", "Is Landon rainy?", 'Is Austrilia always sunny?']
    qs = QuestionCollection.read_csv(question_file).questions
    bp.register_questions(qs)
    
    # models = ['huggyllama/llama-7b', 'meta-llama/Llama-2-7b-hf', 'mosaicml/mpt-7b']
    models = PairwiseBattleArrangement.read_csv(battle_arrangement_file).models
    bp.register_models(models)
    
    bp.arrange_battles(preset=True, preset_save_path=battle_arrangement_file)
    
    bp.gen_model_answers()
    bp.battle()
    pass