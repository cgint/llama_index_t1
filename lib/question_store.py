
import pandas as pd
def store_data(questions_data, questions_data_out_file):
    print(f"Storing collected questions data to {questions_data_out_file} ...")
    questions_data_df = pd.DataFrame(questions_data, columns=["context", "mode", "question", "answer", "time_finished"])
    questions_data_df.to_csv(questions_data_out_file, index=False)