## Prepared Question Sets
- quora 100: manually collected 100 hot quesitons on Quora website.
- google_quora_alpaca_10629: collected question from google trend and quora website, and adds 805 questions from alpaca eval dataset
- google_quora_alpaca_sharegpt_chat1m_21962: adding selected questions from sharegpt and chat1m based on google_quora_alpaca_10629
- google_quora_alpaca_sharegpt_chat1m_20772: remove partial questions from sharegpt and chat1m that are labeled by gpt-4 as questions not suitable for question answering.
- rwq: equal to google_quora_alpaca_sharegpt_chat1m_20772, but remove duplicated questions.
- rwq_200: randomly pick 200 questions from rwq.
