# pylint: disable=invalid-name
# pylint: disable=missing-docstring
# pylint: disable=anomalous-backslash-in-string

# imports {{{
import os
from math import pi
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.mixture import GaussianMixture as GMM
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import umap
# }}}

# functions {{{

def bic_analysis(X_data, file_name, n_comp):
    # Find # of components
    n_components = np.arange(1, n_comp+1)
    models = [GMM(n, covariance_type='full', random_state=seed).fit(X_data) \
              for n in n_components]
    plt.clf()
    plt.close()
    plt.plot(n_components, [m.bic(X_data) for m in models], label='BIC')
    plt.legend(loc='best')
    plt.xlabel('n_components')
    plt.title('BIC for ' + file_name)

    # output
    #plt.show()
    # save
    plt.savefig(file_name + "_bic.png")
    plt.clf()
    plt.close()

# find video frames at means
def extract_mean_frames(data, path, type):
    # type 1 is for gmm1
    # type 2 is for gmm2
    if type == 1:
        num_videos = data.shape[0]
        for i in range(num_videos):
            csv = data['csv_name'].values[i]
            video = csv.split(".")[0] + ".mp4"
            frame = data['frame'].values[i]+1
            frame_name = "mean_frames_1/C" + str(i) + "_mframe.jpg"
            os.system("ffmpeg -i " + path + str(video) + " -vf \"select=eq(n\, " + \
                     str(frame) + ")\" -vframes 1 " + frame_name + " -y")
    elif type == 2:
        num_videos = data.shape[0]
        for i in range(num_videos):
            csv = data['csv_name'].values[i]
            video = csv.split(".")[0] + ".mp4"
            frame = data['frame'].values[i]+1
            frame_name = "mean_frames_2/C" + str(i) + "_mframe.jpg"
            os.system("ffmpeg -i " + path + str(video) + " -vf \"select=eq(n\, " + \
                     str(frame) + ")\" -vframes 1 " + frame_name + " -y")
    else: print("wrong type set for extract_mean_frames")

# find frames closest to mean
def find_mean_frames(data, means, type):
    # type: 1 is for gmm1
    # type: 2 is for gmm2
    if type == 1:
        dfs = []
        num = means.shape[0]
        #print(num)
        for i in range(num):
            data['diff_sum'] = np.abs(data['AU01_r']-means[i][0]) \
                                 + np.abs(data['AU02_r']-means[i][1]) \
                                 + np.abs(data['AU04_r']-means[i][2]) \
                                 + np.abs(data['AU05_r']-means[i][3]) \
                                 + np.abs(data['AU06_r']-means[i][4]) \
                                 + np.abs(data['AU07_r']-means[i][5]) \
                                 + np.abs(data['AU09_r']-means[i][6]) \
                                 + np.abs(data['AU10_r']-means[i][7]) \
                                 + np.abs(data['AU12_r']-means[i][8]) \
                                 + np.abs(data['AU04_r']-means[i][9]) \
                                 + np.abs(data['AU15_r']-means[i][10]) \
                                 + np.abs(data['AU17_r']-means[i][11]) \
                                 + np.abs(data['AU20_r']-means[i][12]) \
                                 + np.abs(data['AU23_r']-means[i][13]) \
                                 + np.abs(data['AU25_r']-means[i][14]) \
                                 + np.abs(data['AU26_r']-means[i][15]) \
                                 + np.abs(data['AU45_r']-means[i][16])

            mean_df = data[data.p_label == i]
            mean_df = mean_df[data.probs >= prob_thresh]

            idx = mean_df[['diff_sum']].idxmin()
            #print(idx)

            mean_df = mean_df.drop(columns=['face_id', 'confidence',
                                            'success']) #, 'diff_sum'])
            mean_df = mean_df.loc[idx]
            dfs.append(mean_df)
            returned_dataframe = pd.concat(dfs)

    if type == 2:
        dfs = []
        num = means.shape[0]
        print(num)
        for i in range(num):
            data['diff_sum'] = np.abs(data['dim1']-means[i][0]) \
                                 + np.abs(data['dim2']-means[i][1]) \

            mean_df = data[data.p_label_umap == i]
            mean_df = mean_df[data.probs_umap >= prob_thresh]

            idx = mean_df[['diff_sum']].idxmin()
            #print(idx)

            #mean_df = mean_df.drop(columns=['face_id', 'confidence',
                                          #  'success']) #, 'diff_sum'])
            mean_df = mean_df[['csv_name', 'frame','dim1', 'dim2',
                'p_label_umap', 'probs_umap', 'diff_sum']]
            mean_df = mean_df.loc[idx]
            dfs.append(mean_df)
            returned_dataframe = pd.concat(dfs)

        else: "wrong type set for find_mean_frames()"

    #print("test_df:\n", mean_df.loc[idx])
    return returned_dataframe

# --- --- find label confidence
def label_confidence(data, type):
    # type 1: GMM1
    # type 2: GMM2 (UMAP)
    conf_dict = {}
    data = data[data.likelihood_diff <= llh_thresh]

    if type == 1:
        data = data[data.probs >= pred_prob_thresh]
        num_values = data.shape[0]
        key = list(data['p_label'].value_counts().index.values)
        value = list(data['p_label'].value_counts().values)

    if type == 2:
        data = data[data.probs_umap >= pred_prob_thresh]
        num_values = data.shape[0]
        key = list(data['p_label_umap'].value_counts().index.values)
        value = list(data['p_label_umap'].value_counts().values)

    for i in range(len(key)):
        conf_dict[key[i]] = (value[i]/num_values).round(2)
    max_conf = 0
    for i in conf_dict:
        if conf_dict[i]>max_conf: max_conf = i
    #print(type(key), type(value))
    return max_conf, conf_dict


# radar chart code adapted from:
# https://python-graph-gallery.com/390-basic-radar-chart/
def make_radar(means, n_comp, data_type, path, gmm_type):
    # 0: use mean data
    # 1: use df data

    if data_type == 0 and gmm_type == 1:
        #print("type is", data_type)
        # Set data
        r_df = pd.DataFrame({
            'group': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8'],
            'AU01': means[:, 0],
            'AU02': means[:, 1],
            'AU04': means[:, 2],
            'AU05': means[:, 3],
            'AU06': means[:, 4],
            'AU07': means[:, 5],
            'AU09': means[:, 6],
            'AU10': means[:, 7],
            'AU12': means[:, 8],
            'AU14': means[:, 9],
            'AU15': means[:, 10],
            'AU17': means[:, 11],
            'AU20': means[:, 12],
            'AU23': means[:, 13],
            'AU25': means[:, 14],
            'AU26': means[:, 15],
            'AU45': means[:, 16]
        })

        #print("df:\n", r_df)
        #print("df shape:", r_df.shape)

        # number of variable
        categories = list(r_df)[1:]
        N = len(categories)

        for i in range(n_comp):
            # We are going to plot each cluster of the data frame.
            # But we need to repeat the first value to close the circular graph:
            values = r_df.loc[i].drop('group').values.flatten().tolist()
            values += values[:1]

            # What will be the angle of each axis in the plot?
            # (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Initialise the spider plot
            ax = plt.subplot(111, polar=True)

            # Draw one axe per variable + add labels labels yet
            plt.xticks(angles[:-1], categories, color='grey', size=8)

            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([1, 2], ["1.0", "2.0"], color="grey", size=7)
            plt.ylim(0, 3)

            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid')

            # Fill area
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title("SIRE Graph for GMM1 cluster " + str(i))
            plt.savefig(path + "/" + r_df.group[i] + "_sire.png")
            ax.clear()

    elif data_type == 1 and gmm_type == 1:
        #print("type is", data_type)
        # Set data
        means = means.copy(deep=True)
        means.reset_index(inplace=True)
        #print("means", means)
        #r_df = means #.copy
        means = means.drop(columns=['p_label', 'probs', 'csv_name', 'frame', 'index', 'diff_sum'])
        group_arr = np.array(['mean0', 'mean1', 'mean2', 'mean3', 'mean4', \
                              'mean5', 'mean6', 'mean7', 'mean8'])
        #r_df['group'] = group_arr #pd.Series(group_arr)
        means.insert(0, "group", group_arr, True)
        #print("means converted", means)
        #print("df:\n", means)
        #print("df shape:", means.shape)

        # number of variable
        categories = list(means)[1:]
        N = len(categories)

        for i in range(n_comp):
            # We are going to plot each cluster of the data frame.
            # But we need to repeat the first value to close the circular graph:
            values = means.loc[i].drop('group').values.flatten().tolist()
            values += values[:1]

            # What will be the angle of each axis in the plot?
            # (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Initialise the spider plot
            ax = plt.subplot(111, polar=True)

            # Draw one axe per variable + add labels labels yet
            plt.xticks(angles[:-1], categories, color='grey', size=8)

            # Draw ylabels
            ax.set_rlabel_position(0)
            plt.yticks([1, 2], ["1.0", "2.0"], color="grey", size=7)
            plt.ylim(0, 3)

            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid')

            # Fill area
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title("SIRE Graph for GMM1 selected frame " + str(i))
            plt.savefig(path + "/" + means.group[i] + "_sire.png")
            ax.clear()

    elif data_type == 0 and gmm_type == 2:
        #print("type is", data_type)
        # Set data
        r_df = pd.DataFrame({
            'group': ['C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7'],
            'dim1': means[:, 0],
            'dim2': means[:, 1],
        })

        #print("df:\n", r_df)
        #print("df shape:", r_df.shape)

        # number of variable
        categories = list(r_df)[1:]
        N = len(categories)

        for i in range(n_comp):
            # We are going to plot each cluster of the data frame.
            # But we need to repeat the first value to close the circular graph:
            values = r_df.loc[i].drop('group').values.flatten().tolist()
            values += values[:1]

            # What will be the angle of each axis in the plot?
            # (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Initialise the spider plot
            ax = plt.subplot(111, polar=True)

            # Draw one axe per variable + add labels labels yet
            plt.xticks(angles[:-1], categories, color='grey', size=8)

            # Draw ylabels
            ax.set_rlabel_position(0)
            #plt.yticks([1, 2], ["1.0", "2.0"], color="grey", size=7)
            #plt.ylim(0, 3)
            plt.yticks(np.linspace(0, 28, num=4), ["0", "7", "14", "21", "28"], color="grey", size=7)
            plt.ylim(0, 28)

            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid')

            # Fill area
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title("SIRE Graph for GMM2 cluster " + str(i))
            plt.savefig(path + "/" + r_df.group[i] + "_sire.png")
            ax.clear()

    elif data_type == 1 and gmm_type == 2:
        #print("type is", data_type)
        # Set data
        means = means.copy(deep=True)
        means.reset_index(inplace=True)
        #print("means", means)
        #r_df = means #.copy
        means = means.drop(columns=['p_label_umap', 'probs_umap', 'csv_name', 'frame', 'index', 'diff_sum'])
        group_arr = np.array(['mean0', 'mean1', 'mean2', 'mean3', 'mean4', \
                              'mean5', 'mean6', 'mean7'])
        #r_df['group'] = group_arr #pd.Series(group_arr)
        means.insert(0, "group", group_arr, True)
        #print("means converted", means)
        #print("df:\n", means)
        #print("df shape:", means.shape)

        # number of variable
        categories = list(means)[1:]
        N = len(categories)

        for i in range(n_comp):
            # We are going to plot each cluster of the data frame.
            # But we need to repeat the first value to close the circular graph:
            values = means.loc[i].drop('group').values.flatten().tolist()
            values += values[:1]

            # What will be the angle of each axis in the plot?
            # (we divide the plot / number of variable)
            angles = [n / float(N) * 2 * pi for n in range(N)]
            angles += angles[:1]

            # Initialise the spider plot
            ax = plt.subplot(111, polar=True)

            # Draw one axe per variable + add labels labels yet
            plt.xticks(angles[:-1], categories, color='grey', size=8)

            # Draw ylabels
            ax.set_rlabel_position(0)
            #plt.yticks([1, 2], ["1.0", "2.0"], color="grey", size=7)
            plt.yticks(np.linspace(0, 28, num=4), ["0", "7", "14", "21", "28"], color="grey", size=7)
            plt.ylim(0, 28)

            # Plot data
            ax.plot(angles, values, linewidth=1, linestyle='solid')

            # Fill area
            ax.fill(angles, values, 'b', alpha=0.1)
            plt.title("SIRE Graph for GMM2 selected frame " + str(i))
            plt.savefig(path + "/" + means.group[i] + "_sire.png")
            ax.clear()


    else: print("Wrong type setting in make_radar")

# Plot Dimension Reduced Data
def plot_umap(emb, n_clusters, predicted_labels, file_name):

    # title conventions
    if file_name == "umap_plot_hdim_lables":
        title = "UMAP Profection of Pheobe\'s AUs using high dim labels \
                and " + str(n_clusters) + " clusters"
    elif file_name == "umap_plot_ldim_lables":
        title = "UMAP Profection of Pheobe\'s AUs using low dim labels \
                and " + str(n_clusters) + " clusters"
    else: title = "UMAP Projection of Pheobe\'s AUs with " + str(n_clusters) + \
               " clusters"

    # scatterplot
    plt.scatter(
        emb[:, 0],
        emb[:, 1],
        c=predicted_labels, cmap='Spectral'
        )
    plt.gca().set_aspect('equal', 'datalim')
    plt.colorbar(boundaries=np.arange(n_clusters+1)-0.5) \
                .set_ticks(np.arange(n_clusters))
    plt.title(title, fontsize=24)
    plt.savefig(file_name + '.png')
    plt.clf()
    plt.close()

def preprocess(csv_path, conf_thresh):

    # get data
    pre_df = pd.read_csv(csv_path)

    # formatting
    csv_name = str(csv_path).split("/")[-1]
    pre_df.columns = pre_df.columns.str.lstrip()
    pre_df.rename(columns={'Unnamed: 0':'frame'}, inplace=True)
    pre_df.insert(0, 'csv_name', csv_name)

    # preprocessing
    pre_df = pre_df[pre_df.success == 1]
    pre_df = pre_df[pre_df.confidence >= conf_thresh]

    return pre_df

# }}}

# Parameters and GMM Init {{{

# parameters
file = "intensity.csv"
train_video_path = "Phoebe_dataset4/Phoebe_dataset4/videos/train/"
test_csv_path = "Phoebe_dataset4/Phoebe_dataset4/single_person_au_intensity/" +\
        "test/"
n = 9
prob_thresh = 0.60
pred_prob_thresh = 0.70
llh_thresh = 14
confidence_thresh = 0.75
seed = 1

# get data
df = pd.read_csv(file)

# remove features
X = df.drop(columns=['csv_name', 'frame', 'face_id', 'confidence', 'success'])

# GMM
# did check seed 7:wrong expressions+same v, 9:c4 wrong, 10: 3 same videos
gmm = GMM(n_components=n, random_state=seed).fit(X)
labels = gmm.predict(X)
df['p_label'] = labels
probs = gmm.predict_proba(X)
df['probs'] = np.array(list(map(max, probs))).round(2)
m = gmm.means_

# make dataframe of video frames closest to the mean, labeled [0-8]
test_df = find_mean_frames(df, m, 1)

# export frames using mean information and put in mean_frames folder
#extract_mean_frames(test_df, train_video_path, 1)

# create sire radar charts & put in sire_graphs folder
#make_radar(gmm.means_, n, 0, "sire_graphs", 1)

# create sire radar chart for selected frames
#make_radar(test_df, n, 1, "sire_graphs", 1)

# }}}

# UMAP GMM {{{

# Umap -- Code adapted from https://umap-learn.readthedocs.io/en/latest/basic_usage.html
# -- Init data
unscaled_data = X.values
#print("data shape:", unscaled_data.shape)
sns.set(style='white', rc={'figure.figsize':(14, 10)})

# -- Show data before transformation
'''
print("creating GMM pairwise plots...")
sns.pairplot(data, hue='label')
plt.show()
'''

# -- Reduce dimensions of data with Umap
#reducer = umap.UMAP()
reducer = umap.UMAP(
        n_neighbors=500,  # 682 for 10; 300-400 looks good
        min_dist=0.60, # 0.8 for 10; 0.5 for 15 comps
        n_components=2,
        random_state=seed
        ) #.fit_transform(unscaled_data)

#scaled_data = StandardScaler().fit_transform(unscaled_data)
#embedding = reducer.fit_transform(scaled_data)
embedding = reducer.fit_transform(unscaled_data)

# -- Plot UMAP to visualize 2D AU data
#plot_umap(embedding, 9, df.p_label, "umap_plot_hdim_labels")

# Reformat UMAP data for GMM
df['dim1'] = embedding[:, 0]
df['dim2'] = embedding[:, 1]
X2 = df[['dim1', 'dim2']]

# Check BIC scores
#bic_analysis(X, "original", 30)
#bic_analysis(X2, "UMAP", 30)
n_umap = 8

# Re-created GMM
gmm_umap = GMM(n_components=n_umap, random_state=seed).fit(X2)
labels_umap = gmm_umap.predict(X2)
df['p_label_umap'] = labels_umap
probs_umap = gmm_umap.predict_proba(X2)
df['probs_umap'] = np.array(list(map(max, probs_umap))).round(2)
m_umap = gmm_umap.means_

#plot_umap(embedding, n_umap, df.p_label_umap, "umap_plot_ldim_labels")

# Compare High-Dimensional Labels to Low-Dimensional Labels
#print("adjusted rand score:", adjusted_rand_score(labels, labels_umap))
#print("adjusted mutual score:", adjusted_mutual_info_score(labels, labels_umap))

'''
Using the adjusted rand and mutual information scores, it seems obvious that
clustering on this lower dimensional space did not go so well.
from [the page on clustering] it seems as though distances were not preserved
'''

#print("data:\n", df)
#print("means:\n", m_umap)

# make dataframe of video frames closest to the mean, labeled [0-7]
#mumap_df = find_mean_frames(df, m_umap, 2)

# export frames using mean information and put in mean_frames folder
#extract_mean_frames(mumap_df, train_video_path, 2)

# create new sire radar charts for 2-D data & put in sire_graphs_2 folder
#make_radar(m_umap, n_umap, 0, "sire_graphs_2", 2)

# create new sire charts for new selected frames & put in sire_graphs_2 folder
#make_radar(mumap_df, n_umap, 1, "sire_graphs_2", 2)

# }}}

# Classification of videos 51, 52, and 53
'''
GMM1 Classification
===========================
'''
gmm1_labels = {
0:"Joy/Laughter",
1:"Neutral/Embarassed",
2:"Sad",
3:"Happy/Bashful",
4:"Disgust",
5:"Neutral/Annoyed/Upset",
6:"Surpise/Sad",
7:"Happy (Low Activation)",
8:"Sadness/Fear/Anxious"
}
'''
GMM2 Classification
===========================
'''
gmm2_labels = {
0:"excitement/surprise",
1:"neutral or fear",
2:"disgust",
3:"sad",
4:"contempt",
5:"upset/angry",
6:"happy, low activation or embaressment",
7:"happy, high activation"
}

# --- init & Format
df_51 = preprocess(test_csv_path + "51.csv", confidence_thresh)
df_52 = preprocess(test_csv_path + "52.csv", confidence_thresh)
df_53 = preprocess(test_csv_path + "53.csv", confidence_thresh)

# --- GMM1
# --- --- remove features
X_51 = df_51.drop(columns=['csv_name', 'frame', 'face_id', 'confidence',
                           'success'])
X_52 = df_52.drop(columns=['csv_name', 'frame', 'face_id', 'confidence',
                           'success'])
X_53 = df_53.drop(columns=['csv_name', 'frame', 'face_id', 'confidence',
                           'success'])

# --- --- Predict & Label DFs
labels_51 = gmm.predict(X_51)
likelihood_51 = gmm.score_samples(X_51)
df_51['p_label'] = labels_51
df_51['likelihood_diff'] = abs(gmm.score(X_51) - likelihood_51).round(2)
probs_51 = gmm.predict_proba(X_51)
df_51['probs'] = np.array(list(map(max, probs_51))).round(2)

labels_52 = gmm.predict(X_52)
likelihood_52 = gmm.score_samples(X_52)
df_52['p_label'] = labels_52
df_52['likelihood_diff'] = abs(gmm.score(X_52) - likelihood_52).round(2)
probs_52 = gmm.predict_proba(X_52)
df_52['probs'] = np.array(list(map(max, probs_52))).round(2)

labels_53 = gmm.predict(X_53)
likelihood_53 = gmm.score_samples(X_53)
df_53['p_label'] = labels_53
df_53['likelihood_diff'] = abs(gmm.score(X_53) - likelihood_53).round(2)
probs_53 = gmm.predict_proba(X_53)
df_53['probs'] = np.array(list(map(max, probs_53))).round(2)

# --- --- Calculate confidence level of each label

pred_51_1, conf_51_1 = label_confidence(df_51, 1)
pred_52_1, conf_52_1 = label_confidence(df_52, 1)
pred_53_1, conf_53_1 = label_confidence(df_53, 1)
print("video 51 predicted as", gmm1_labels[pred_51_1], "with model 1 and confidence scores:", conf_51_1)
print("video 52 predicted as", gmm1_labels[pred_52_1], "with model 1 and confidence scores:", conf_52_1)
print("video 53 predicted as", gmm1_labels[pred_53_1], "with model 1 and confidence scores:", conf_53_1)

# GMM2: Umap ============================================================

# --- Init data
unscaled_data_51 = X_51.values
unscaled_data_52 = X_52.values
unscaled_data_53 = X_53.values

# --- Reduce dimensions of data with Umap model
reducer_2 = umap.UMAP(
        n_neighbors=10,  # 682 for 10; 300-400 looks good
        min_dist=0.60, # 0.8 for 10; 0.5 for 15 comps
        n_components=2,
        random_state=seed
        )
"""
reducer_2 = umap.UMAP()
"""
embedding_51 = reducer_2.fit_transform(unscaled_data_51)
embedding_52 = reducer_2.fit_transform(unscaled_data_52)
embedding_53 = reducer_2.fit_transform(unscaled_data_53)

# --- Reformat UMAP data for GMM
df_51['dim1'] = embedding_51[:, 0]
df_51['dim2'] = embedding_51[:, 1]
X2_51 = df_51[['dim1', 'dim2']]

df_52['dim1'] = embedding_52[:, 0]
df_52['dim2'] = embedding_52[:, 1]
X2_52 = df_52[['dim1', 'dim2']]

df_53['dim1'] = embedding_53[:, 0]
df_53['dim2'] = embedding_53[:, 1]
X2_53 = df_53[['dim1', 'dim2']]

# --- Re-created GMM

#gmm_umap_51 = GMM(n_components=n_umap, random_state=seed).fit(X2)
labels_umap_51 = gmm_umap.predict(X2_51)
df_51['p_label_umap'] = labels_umap_51
probs_umap_51 = gmm_umap.predict_proba(X2_51)
df_51['probs_umap'] = np.array(list(map(max, probs_umap_51))).round(2)

labels_umap_52 = gmm_umap.predict(X2_52)
df_52['p_label_umap'] = labels_umap_52
probs_umap_52 = gmm_umap.predict_proba(X2_52)
df_52['probs_umap'] = np.array(list(map(max, probs_umap_52))).round(2)

labels_umap_53 = gmm_umap.predict(X2_53)
df_53['p_label_umap'] = labels_umap_53
probs_umap_53 = gmm_umap.predict_proba(X2_53)
df_53['probs_umap'] = np.array(list(map(max, probs_umap_53))).round(2)

# --- --- Calculate confidence level of each label

pred_51_2, conf_51_2 = label_confidence(df_51, 2)
pred_52_2, conf_52_2 = label_confidence(df_52, 2)
pred_53_2, conf_53_2 = label_confidence(df_53, 2)
print("video 51 predicted as", gmm2_labels[pred_51_2], "with model 2 and confidence scores:", conf_51_2)
print("video 52 predicted as", gmm2_labels[pred_52_2], "with model 2 and confidence scores:", conf_52_2)
print("video 53 predicted as", gmm2_labels[pred_53_2], "with model 2 and confidence scores:", conf_53_2)


# output {{{
#print("gmm2 df:\n", mumap_df)
#print("embedding:", embedding)
#print("embedding shape:", embedding.shape)
#print(unscaled_data.shape)
'''
print("Labels:", labels, "\n")
print("Score Samples:\n", gmm.score_samples(X), "\n")
print("Probabilities:\n", probs.round(2), "\n")
print("Weights:\n", gmm.weights_, "\n")
print("Score:", gmm.score(X), "\n")
#print("Mean to find:\n", gmm.means_[0].round(2), "\n")
#print("mean frames:\n", test_df)
'''

# }}}

# export dataframe
#df.to_csv('pheobe_data.csv')
df_51.to_csv('data_51.csv')
df_52.to_csv('data_52.csv')
df_53.to_csv('data_53.csv')
