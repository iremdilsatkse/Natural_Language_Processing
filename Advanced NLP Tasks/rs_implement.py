from surprise import Dataset, KNNBasic, accuracy
from suprise.model_selection import train_test_split

data = Dataset.load_builtin("ml-100k")

trainset, testset = train_test_split(data, test_size = 0.2)

model_options = {
    "name": "cosine",
    "user_based": True
}

model = KNNBasic(sim_options = model_options)
model.fit(trainset)

prediction = model.test(testset)
accuracy.rmse(prediction)

def get_pre(prediction, n = 10):
    top_n = {}

    for u_id, i_id, true_r, est, _ in prediction:
        if not top_n.get(u_id):
            top_n[u_id] = []
        top_n[u_id].append((i_id, est))

    for u_id, user_ratings in top_n.items():
        user_ratings.sort(key=lambda x: x[1], reverse=True)
        top_n[u_id] = user_ratings[:n]

    return top_n

n = 5
top_n = get_pre(prediction, n)

user_id = "2"
print(f"top {n} recommendation for user {user_id}")
for item_id, rating in top_n[user_id]:
    print(f"item id: {item_id}, score: {rating}")









