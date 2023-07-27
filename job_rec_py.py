import time
import pandas as pd
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
t0 = time.time()

class JobRec():
    def __init__(self):
        df = pd.read_csv('jobs_data.csv')

        df_dropped = self.preprocess_df(df)
        
        tfidf = TfidfVectorizer(stop_words='english')

        df = df_dropped.fillna('')

        df['title'] = df['title'].fillna('')

        tfidf_matrix_title = tfidf.fit_transform(df['title'])

        cosine_sim = linear_kernel(tfidf_matrix_title, tfidf_matrix_title)

        df = df.reset_index()

        indices = pd.Series(df.index, index=df['title']).drop_duplicates()
        
        self.df_dropped = df_dropped
        self.indices = indices
        self.cosine_sim = cosine_sim
        self.titles = df.title.drop_duplicates().dropna()
        self.df = df



    def preprocess_jf(self, df):
        vals_jf = []

        for val in df["jobFunction"]:
            x = ast.literal_eval(val)
            vals_jf.append(x)

        jf_df = pd.DataFrame(data=vals_jf)
        jf_df.columns = ['jf_1', 'jf_2', 'jf_3']

        df_no_jf = df.drop(columns='jobFunction') 

        df_new = pd.concat([df_no_jf, jf_df], axis=1)
        df_new.sort_values(by='title')
        
        return df_new


    def preprocess_ind(self, df):
        vals_ind = []

        for val in df["industry"]:
            x = ast.literal_eval(val)
            vals_ind.append(x)

        ind_df = pd.DataFrame(data=vals_ind)
        
        ind_df.columns = ['ind_1', 'ind_2', 'ind_3']
        df_no_ind = df.drop(columns='industry') 
        df_final = pd.concat([df_no_ind, ind_df], axis=1)
        
        return df_final


    def preprocess_df(self, df):

        df_pp = self.preprocess_jf(df)
        df_final = self.preprocess_ind(df_pp)

        df_final_drop = df_final.drop_duplicates()
        df2 = df_final_drop.fillna('')
        
        return df2

    def get_recommendations(self, df, title):

        idx = self.indices[title]

        sim_scores = list(enumerate(self.cosine_sim[idx]))
 
        sim_scores = sorted(sim_scores, reverse=True)

        sim_scores = sim_scores[0:10]

        rec_jobs_indices = [i[0] for i in sim_scores]

        recommendations = self.df_dropped.iloc[rec_jobs_indices, :]
        
        return recommendations

    def get_recs(self, df, title):
        rec_dict = {} 
        recommendations = self.get_recommendations(df, title)
        recommendations = recommendations.drop_duplicates()

        recommended_job_titles = [recommendations['title'].drop_duplicates().tolist()]

        title_list = sum(recommended_job_titles, [title])

        title_list = list(filter(None, title_list))

        rec_dict['titles'] = title_list

        recommended_job_functions = [recommendations[fun].drop_duplicates().tolist() for fun in ['jf_1', 'jf_2', 'jf_3']] 

        fun_list = sum(recommended_job_functions, [])
        fun_list = list(set(filter(None, fun_list)))   
        rec_dict['functions'] = fun_list
        
        recommended_job_industries = [recommendations[ind].drop_duplicates().tolist() for ind in ['ind_1', 'ind_2', 'ind_3']] 
        ind_list = sum(recommended_job_industries, [])
        ind_list = list(set(filter(None, ind_list)))
        rec_dict['industries'] = ind_list

        return rec_dict


    def recommend(self, substring):
        recommendations_list = []

        suggested_titles = [title for title in self.titles if title.lower().find(substring.lower()) != -1][:5]
        print(f"SUGGESTED TITLES: {suggested_titles}")

        for suggested_title in suggested_titles:
            recommendation_dict = self.get_recs(self.df, suggested_title)
            recommendations_list.append(recommendation_dict)
            print(f"RECOMMENDATIONS FOR {suggested_title}:\n {recommendation_dict}")
            print("\n--------------------------------\n")
        
        return suggested_titles, recommendations_list



# CALCULATING ACCURACY

df = pd.read_csv('jobs_data.csv')

train_data, test_data = train_test_split(df, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words='english')

train_title_matrix = tfidf.fit_transform(train_data['title'])

test_title_matrix = tfidf.transform(test_data['title'])

cosine_sim = linear_kernel(test_title_matrix, train_title_matrix)

similar_jobs_indices = cosine_sim.argmax(axis=1)

predicted_titles = train_data.iloc[similar_jobs_indices]['title'].values

actual_titles = test_data['title'].values

accuracy = accuracy_score(actual_titles, predicted_titles)

print(f"Accuracy: {accuracy:.2f}")


if __name__ == '__main__':
    t0 = time.time()
    JR = JobRec()

    JR.recommend(substring='data')

    task_time = time.time() - t0
    print(f"This task took {task_time:.3f}sec")