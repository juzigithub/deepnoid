


svc = SVC(kernel="linear", C=1)
from sklearn.linear_model import LogisticRegression
# model = sc()
rfe = RFE(estimator=svc, n_features_to_select=300, step=1,verbose=True)
rfe = rfe.fit(x, y)
# ranking = rfe.ranking_.reshape(digits.images[0].shape)
print(rfe.support_)
print(rfe.ranking_ )
feature_list =[]
for i in range(len(x)):
   if rfe.ranking_[i] <= 300:
       print(df.columns[i+1])
       feature_list.append(df.columns[i+1])
print(feature_list)
feature_list = np.asarray(feature_list)
np.savetxt("ranking_.csv", rfe.ranking_, delimiter=",")

df = pd.DataFrame(feature_list)
df.to_csv("feature_list.csv")
df2 = pd.DataFrame(rfe.ranking_)
df.to_csv("ranking_2.csv")
# print(df.columns if rfe.ranking_ == 1)
np.savetxt("foo.csv", feature_list, delimiter=",")