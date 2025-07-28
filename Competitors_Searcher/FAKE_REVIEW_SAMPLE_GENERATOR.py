import pandas as pd
import random

# Function to generate random reviews
def generate_random_reviews(num_reviews):
    companies = ["serryfacile.it", "amazon.com", "amazon.it", "costco.com"]
    sentiments = ["terrible", "bad", "average", "good", "excellent"]
    
    reviews = []
    scores = []
    sentiment_score_map = {
        "terrible": (0, 2),
        "bad": (2, 4),
        "average": (4, 6),
        "good": (6, 8),
        "excellent": (8, 10),
    }

    for _ in range(num_reviews):
        company = random.choice(companies)
        sentiment = random.choice(sentiments)
        
        # Get the score range for the selected sentiment
        score_min, score_max = sentiment_score_map[sentiment]
        score = random.uniform(score_min, score_max)  # Random score within the sentiment's range
        
        # Create a random review based on sentiment
        if sentiment == "terrible":
            review = f"I had a terrible experience with {company}. Absolutely {sentiment}!"
        elif sentiment == "bad":
            review = f"The service from {company} was {sentiment}. Not impressed."
        elif sentiment == "average":
            review = f"{company} was just average. Nothing special to report."
        elif sentiment == "good":
            review = f"I had a good experience with {company}. I would recommend it!"
        elif sentiment == "excellent":
            review = f"An excellent experience with {company}. Highly satisfied!"
        
        reviews.append(review)
        scores.append(score)
    
    return pd.DataFrame({
        'review': reviews,
        'score': scores,
        'company': [random.choice(companies) for _ in range(num_reviews)]
    })

# Generate a dataset with 1000 random reviews
num_reviews = 10000000
random_reviews_df = generate_random_reviews(num_reviews)

# Save the dataframe to CSV
random_reviews_df.to_csv('trustpilot_reviews.csv', index=False)

print(f"{num_reviews} random reviews generated and saved to 'trustpilot_reviews.csv'")
