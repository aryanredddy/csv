import os
import pandas as pd
import matplotlib.pyplot as plt
import openai
import pandas as pd

data = {
    'Name': ['Aryan', 'Prashanth', 'Seenu'],
    'Age': [25, 26, 32],
    'Salary': [50000, 60000, 45000]
}

df = pd.DataFrame(data)
df.to_csv('data.csv', index=False)



def l_csv(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"The file {file_path} was not found.")
    return pd.read_csv(file_path)

def basic_stat(df):
    numeric_df = df.select_dtypes(include='number')

    stats = {
        'mean': numeric_df.mean(),
        'median': numeric_df.median(),
        'mode': numeric_df.mode().iloc[0], 
        'std_dev': numeric_df.std(),
        'correlation': numeric_df.corr()
    }
    return stats


def gen_plot(df, columns):
    if len(columns) > 0:
        df[columns].hist()
        plt.suptitle('Histogram')
        plt.show()

    if len(columns) == 2:
        df.plot.scatter(x=columns[0], y=columns[1])
        plt.title('Scatter Plot')
        plt.show()

    if len(columns) > 0:
        df[columns].plot()
        plt.title('Line Plot')
        plt.show()

openai.api_key = os.getenv('OPENAI_API_KEY', 'your-api-key')

def ask_llm(prompt):
    response = openai.Completion.create(
        engine="text-davinci-003",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

def analyze_llm(data_description):
    prompt = f"Here is a dataset description:\n{data_description}\nCan you provide insights?"
    return ask_llm(prompt)

class CSVAnalyzer:
    def __init__(self, file_path):
        self.df = l_csv(file_path)
    
    def analyze(self):
        stats = basic_stat(self.df)
        print("Basic Statistics:", stats)
        
        gen_plot(self.df, self.df.columns)
        
        description = self.df.describe().to_string()
        llm_analysis = analyze_llm(description)
        print("LLM Analysis:", llm_analysis)

if __name__ == "__main__":
    analyzer = CSVAnalyzer('data.csv')
    analyzer.analyze()
