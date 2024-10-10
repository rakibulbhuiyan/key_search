import pandas as pd
import re
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Preprocessing function
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def preprocess_description(description):
    description = description.lower()
    description = re.sub('[^a-zA-Z]', ' ', description)
    words = description.split()
    words = [ps.stem(word) for word in words if word not in stop_words]
    return ' '.join(words)

# IT-related keywords
it_keywords = ['python', 'java', 'c++', 'machine learning', 'data science', 
               'sql', 'hadoop', 'spark', 'r', 'javascript', 'php', 'deep learning']

class JobDatasetAnalysisAPIView(APIView):
    def get(self, request):
        try:
            # Load the dataset from JobsDataset.csv
            dataset = pd.read_csv('JobsDataset.csv')
            
            # Preprocess all descriptions
            dataset['Description_Clean'] = dataset['Description' ].apply(preprocess_description)
            
            # Perform TF-IDF transformation
            vectorizer = TfidfVectorizer(vocabulary=it_keywords)
            X = vectorizer.fit_transform(dataset['Description_Clean'])
            tfidf_df = pd.DataFrame(X.toarray(), columns=vectorizer.get_feature_names_out())
            
            # Merge TF-IDF data with the dataset
            dataset_tfidf = pd.concat([dataset, tfidf_df], axis=1)

            # Process each row to calculate keyword percentages and skillsets
            job_results = []
            for index, row in dataset_tfidf.iterrows():
                keyword_count = row[it_keywords].sum()
                total_keywords = len(it_keywords)
                weight_percentage = (keyword_count / total_keywords) * 100
                
                # Calculate individual keyword percentages
                keyword_percentages = {keyword: (row[keyword] / total_keywords) * 100 for keyword in it_keywords}
                
                # Determine primary and secondary skillsets
                keyword_scores = {keyword: row[keyword] for keyword in it_keywords}
                sorted_keywords = sorted(keyword_scores.items(), key=lambda x: x[1], reverse=True)
                primary_skillset = sorted_keywords[0][0] if sorted_keywords[0][1] > 0 else 'None'
                secondary_skillset = [k[0] for k in sorted_keywords[1:] if k[1] > 0]
                secondary_skillset = secondary_skillset if secondary_skillset else ['None']

                job_results.append({
                    "description": row['Description'],
                    "keyword_percentages": keyword_percentages,
                    "primary_skillset": primary_skillset,
                    "secondary_skillset": secondary_skillset
                })

            # Return the analysis for all job descriptions
            return Response(job_results, status=status.HTTP_200_OK)

        except FileNotFoundError:
            return Response({"error": "JobsDataset.csv not found"}, status=status.HTTP_404_NOT_FOUND)
