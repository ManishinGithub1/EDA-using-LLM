import gradio as gr
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import ollama

def eda_analysis(file_path):
    df=pd.read_csv(file_path)

    for col in df.select_dtypes(include=['number']).columns:
        df[col].fillna(df[col].median(),inplace=True)

    for col in df.select_dtypes(include=['object']).columns:
        df[col].fillna(df[col].mode()[0],inplace=True)
        


    summary=df.describe(include='all').to_string()

    missing_values=df.isnull().sum().to_string()

    insights=generate_ai_insights(summary)

    plot_path=generate_visualizations(df)

    return f"\n Data Summary: \n {summary} \n Missing Values: \n {missing_values} \n AI Insights: \n {insights} \n Visualizations: \n",plot_path

def generate_ai_insights(df_summary):
    prompt=f'Analyze the dataset and provide insights: \n\n {df_summary}'
    response=ollama.chat(model="mistral", messages=[{"role":"user","content":prompt}])
    print(response)  # Print the response to understand its structure
    return response['message']['content']  # Adjust the key access based on the actual response structure

def generate_visualizations(df):
    plot_path=[]

    for col in df.select_dtypes(include=['number']).columns:
        plt.figure(figsize=(10,6))
        sns.histplot(df[col],bins=50,kde=True,color='blue')
        plt.title(f'Distribution of {col}')
        path=f'{col} distribution.png'
        plt.savefig(path)
        plot_path.append(path)
        plt.close()

#correlation heatmap
    numeric_df=df.select_dtypes(include=['number'])    
    if not numeric_df.empty:
    
        plt.figure(figsize=(8,6))
        sns.heatmap(numeric_df.corr(),annot=True,cmap='coolwarm')
        plt.title('Correlation Heatmap')
        path='correlation_heatmap.png'
        plt.savefig(path)
        plot_path.append(path)
        plt.close()

    return plot_path

app=gr.Interface(fn=eda_analysis,
                inputs=gr.File(type="filepath"),
                outputs=[gr.Textbox(label="Eda Report"),
                gr.Gallery(label="Data Visualizations")],
                title="EDA and LLLM Integration",
                description="Analyze the dataset and provide insights, missing values, and visualizations")

app.launch()

