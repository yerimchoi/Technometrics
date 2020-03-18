import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from scipy.spatial.distance import cdist

# confusion = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Results/confusion_matrix_list_over650.csv', header = None))
# result = pd.DataFrame(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Results/stacked_model_result_over650.csv'))
#
# relu    = confusion[:, :25 * 26]
# sigmoid = confusion[:, 25 * 26:25 * 26 * 2]
# linear  = confusion[:, 25 * 26 * 2:25 * 26 * 3]
# softmax = confusion[:, 25 * 26 * 3 :25 * 26 * 4]
#
# relu_yhat = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Results/argmax_relu_yhat_over650.csv', header = None))
# sigmoid_yhat = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Results/argmax_sigmoid_yhat_over650.csv', header = None))
# linear_yhat = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Results/argmax_linear_yhat_over650.csv', header = None))
# softmax_yhat = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/Results/argmax_softmax_yhat_over650.csv', header = None))


###################################################################################
# yhat
###################################################################################

class DiversityMeasure:
    def __init__(self, matrix_A, matrix_B):
        # matrix_A, matrix_B 는 yhat == y면 1, yhat!= y면 0인 matrix.
        self.matrix_A = matrix_A
        self.matrix_B = matrix_B

    def mahalanobis(self):
        mah_yhat_diag = []
        mah_yhat_all  = []
        for i in range(len(self.matrix_A)):
            row_A = self.matrix_A[i].reshape(-1, 1)      # (150, 1)
            row_B = self.matrix_B[i].reshape(-1, 1)
            if (row_A == row_B).all() :
                mah_yhat_diag.append(0)
                mah_yhat_all.append(0)
            else:
                mah_diag = np.sum(np.diag(cdist(row_A, row_B, 'mahalanobis')))
                mah_all = np.sum(cdist(row_A, row_B, 'mahalanobis'))
                mah_yhat_diag.append(mah_diag)
                mah_yhat_all.append(mah_all)
        mah_yhat_diag = np.array(mah_yhat_diag)
        mah_yhat_all = np.array(mah_yhat_all)
        return mah_yhat_diag, mah_yhat_all

    def euclidean(self):
        euc = np.sum(np.square(self.matrix_A - self.matrix_B) , axis = 1)    # 행끼리 더함
        # print(euc.shape)
        return euc


def transform_yhat(matrix):
    letters = pd.read_csv("C:/Users/lvaid/skku/Technometrics/Data/Letter Recognition.csv")
    test_y = np.array(letters[15000:]['letter'])
    encoder = LabelEncoder()
    encoder.fit(test_y)
    test_y = encoder.transform(test_y)

    length = len(matrix)
    letter_matrix = np.array([test_y] * length)

    after_matrix = letter_matrix - matrix
    after_matrix[after_matrix != 0] = -1
    after_matrix[after_matrix >= 0] = 1
    after_matrix[after_matrix < 0] = 0
    return after_matrix

def yhat_distances(relu_yhat, sigmoid_yhat, linear_yhat, softmax_yhat, result):
    # relu_yhat = transform_yhat(relu_yhat)   # (1000, 5000)
    # sigmoid_yhat = transform_yhat(sigmoid_yhat)
    # linear_yhat = transform_yhat(linear_yhat)
    # softmax_yhat = transform_yhat(softmax_yhat)
    # print(result.shape)     # (1000, 3)

    relu_sigmoid = DiversityMeasure(relu_yhat, sigmoid_yhat)
    sigmoid_linear = DiversityMeasure(sigmoid_yhat, linear_yhat)
    linear_softmax = DiversityMeasure(linear_yhat, softmax_yhat)
    relu_linear = DiversityMeasure(relu_yhat, linear_yhat)
    relu_softmax = DiversityMeasure(relu_yhat, softmax_yhat)
    sigmoid_softmax = DiversityMeasure(sigmoid_yhat, softmax_yhat)

    euc_1 = relu_sigmoid.euclidean() + sigmoid_linear.euclidean() + \
            linear_softmax.euclidean() + relu_softmax.euclidean()
    euc_1 = euc_1.reshape(-1, 1)

    euc_2 = relu_sigmoid.euclidean() + relu_linear.euclidean() + \
            relu_softmax.euclidean() + sigmoid_linear.euclidean() + \
            sigmoid_softmax.euclidean() + linear_softmax.euclidean()
    euc_2 = euc_2.reshape(-1, 1)

    mah_relu_sigmoid_diag, mah_relu_sigmoid_all         = relu_sigmoid.mahalanobis()
    mah_relu_linear_diag, mah_relu_linear_all           = relu_linear.mahalanobis()
    mah_relu_softmax_diag, mah_relu_softmax_all         = relu_softmax.mahalanobis()
    mah_sigmoid_linear_diag, mah_sigmoid_linear_all     = sigmoid_linear.mahalanobis()
    mah_sigmoid_softmax_diag, mah_sigmoid_softmax_all   = sigmoid_softmax.mahalanobis()
    mah_linear_softmax_diag, mah_linear_softmax_all     = linear_softmax.mahalanobis()

    mah_1 = mah_relu_sigmoid_all + mah_sigmoid_linear_all + mah_linear_softmax_all + mah_relu_softmax_all
    mah_1 = mah_1.reshape(-1, 1)

    mah_2 = mah_relu_sigmoid_all + mah_relu_linear_all + mah_relu_softmax_all + \
            mah_sigmoid_linear_all + mah_sigmoid_softmax_all + mah_linear_softmax_all
    mah_2 = mah_2.reshape(-1, 1)

    yhat_matrix = np.hstack([euc_1, euc_2, mah_1, mah_2, result])
    return yhat_matrix

def yhat_correlation(dist_matrix):
    euc_1 = dist_matrix[:,0]
    euc_2 = dist_matrix[:,1]
    mah_1 = dist_matrix[:,2]
    mah_2 = dist_matrix[:,3]
    acc = dist_matrix[:,4]
    auc = dist_matrix[:,5]
    recall = dist_matrix[:,6]

    euc_1_acc = np.corrcoef(euc_1, acc)[0][1]
    euc_2_acc = np.corrcoef(euc_2, acc)[0][1]
    mah_1_acc = np.corrcoef(mah_1, acc)[0][1]
    mah_2_acc = np.corrcoef(mah_2, acc)[0][1]
    mah_1_auc = np.corrcoef(mah_1, auc)[0][1]
    mah_2_auc = np.corrcoef(mah_2, auc)[0][1]
    mah_1_recall = np.corrcoef(mah_1, recall)[0][1]
    mah_2_recall = np.corrcoef(mah_2, recall)[0][1]

    print("euc:", euc_1_acc, euc_2_acc )
    print("mah_acc :", mah_1_acc, mah_2_acc)
    print("mah_auc :", mah_1_auc, mah_2_auc)
    print("mah_recall :", mah_1_recall, mah_2_recall)
    return euc_1_acc, euc_2_acc, mah_1_acc, mah_2_acc, mah_1_auc, mah_2_auc, mah_1_recall, mah_2_recall

###################################################################################
# pairwise diversity measure
###################################################################################

def pairmatrix(matrix_A, matrix_B):
    # matrix_A == classifer 1, matrix_B == classifier 2
    correct = np.where(matrix_A == matrix_B, 1, 10)
    false = np.where(matrix_A != matrix_B, 1, -10)

    num1 = np.where(matrix_A == 1, 1, 0)
    num4 = np.where(matrix_A == 0, 1, 0)

    n11 = np.sum(np.where(correct == num1, 1, 0), axis=1)
    n00 = np.sum(np.where(correct == num4, 1, 0), axis=1)
    n10 = np.sum(np.where(false == num1, 1, 0), axis=1)
    n01 = np.sum(np.where(false == num4, 1, 0), axis=1)

    shape = n11.shape[0]
    q = ((n11 * n00 - n01 * n10) / (n11 * n00 + n01 * n10))   # shape = (100, )
    lo = ((n11 * n00 - n01 * n10) / np.sqrt((n11 + n10) * (n01 + n00) * (n11 + n01) * (n10 + n00)))
    dis = ((n01 + n10) / (n11 + n10 + n01 + n00))
    df = (n00 / (n11 + n10 + n01 + n00))
    return q, lo, dis, df

def average_measure(m_list):
    length = len(m_list)
    sum = m_list[0]
    for i in range(length):
        if i == length - 1:
            break
        sum = sum + m_list[i + 1]
    average = 2 * sum / length * (length - 1)
    # print(average.shape)    # (100, )
    return average

def pairwise(relu, sigmoid, linear, softmax):
    q_rs, lo_rs, dis_rs, df_rs = pairmatrix(relu, sigmoid)    # (100, 1)
    q_rl, lo_rl, dis_rl, df_rl = pairmatrix(relu, linear)
    q_rx, lo_rx, dis_rx, df_rx = pairmatrix(relu, softmax)
    q_sl, lo_sl, dis_sl, df_sl = pairmatrix(sigmoid, linear)
    q_sx, lo_sx, dis_sx, df_sx = pairmatrix(sigmoid, softmax)
    q_lx, lo_lx, dis_lx, df_lx = pairmatrix(linear, softmax)

    q_list = [q_rs, q_rl, q_rx, q_sl, q_sx, q_lx]
    q_av = average_measure(q_list)
    lo_list = [lo_rs, lo_rl, lo_rx, lo_sl, lo_sx, lo_lx]
    lo_av = average_measure(lo_list)
    dis_list = [dis_rs, dis_rl, dis_rx, dis_sl, dis_sx, dis_lx]
    dis_av = average_measure(dis_list)
    df_list = [df_rs, df_rl, df_rx, df_sl, df_sx, df_lx]
    df_av = average_measure(df_list)
    return q_av, lo_av, dis_av, df_av

def pariwise_correlation(q, lo, dis, df, result):
    acc = result[:,0]
    auc = result[:,1]
    recall = result[:,2]

    that = [q, lo, dis, df]
    measure = [acc, auc, recall]
    cor_list = list_to_cor(that, measure)
    return cor_list

def list_to_cor(that_list, measure_list):
    cor_list = []   # q 4개, lo 4개 ...
    for i in range(len(measure_list)):
        for j in range(len(that_list)):
            temp = np.corrcoef(that_list[j], measure_list[i])[0][1]
            cor_list.append(temp)
    return cor_list

###################################################################################
# confusion matrix
###################################################################################


if __name__ == "__main__":
    path_dir = 'C:/Users/lvaid/skku/Technometrics/Data/Dataset/experiment'
    file_list = os.listdir(path_dir)
    file_list.sort()
    correlation_result_list = []


    for file in file_list:
        print(file)

        # load the result files
        # yhat result
        relu_yhat = np.array(pd.read_csv(
            'C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' + str(
                file) + '_argmax_relu_yhat.csv', header = None))
        sigmoid_yhat = np.array(pd.read_csv(
            'C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' + str(
                file) + '_argmax_sigmoid_yhat.csv', header = None))
        linear_yhat = np.array(pd.read_csv(
            'C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' + str(
                file) + '_argmax_linear_yhat.csv', header = None))
        softmax_yhat = np.array(pd.read_csv(
            'C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' + str(
                file) + '_argmax_softmax_yhat.csv', header = None))

        # correct matrix result
        relu_correct = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' +
                                            str(file) + 'correct_relu.csv', header = None))
        sigmoid_correct = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' +
                                            str(file) + 'correct_sigmoid.csv', header = None))
        linear_correct = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' +
                                            str(file) + 'correct_linear.csv', header = None))
        softmax_correct = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' +
                                            str(file) + 'correct_softmax.csv', header = None))

        # accuracy & confusion matrix
        result = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' +
                             str(file) + 'binary_stacked_model_result.csv'))    # (100, 3)
        confusion = np.array(pd.read_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/' +
                                str(file) + 'Binary_confusion_matrix_list.csv', header = None))

        # Tranlating the results

        # 1 yhat
        # distance_matrix = yhat_distances(relu_yhat, sigmoid_yhat, linear_yhat, softmax_yhat, result)
        # euc_1_acc, euc_2_acc, mah_1_acc, mah_2_acc, mah_1_auc, mah_2_auc, mah_1_recall, mah_2_recall = yhat_correlation(distance_matrix)

        # 2 pairwise diversity
        q, lo, dis, df = pairwise(relu_correct, sigmoid_correct, linear_correct, softmax_correct)
        diversity_measure = []
        print(len(q))
        for i in range(len(q)):
            p_div = [q[i], lo[i], dis[i], df[i]]
            diversity_measure.append(p_div)
            diversity_measure_dt = pd.DataFrame(diversity_measure)
            diversity_measure_dt.to_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/diversity_value' + \
                str(file) + '.csv')
        # d_list = pariwise_correlation(q, lo, dis, df, result)

        # 3 confusion matrix
        # Save the results
        # list = [file, euc_1_acc, euc_2_acc, mah_1_acc, mah_2_acc, mah_1_auc, mah_2_auc, mah_1_recall, mah_2_recall]
        # for i in range(len(d_list)):
        #     list.append(d_list[i])
        # correlation_result_list.append(list)
        # dt = pd.DataFrame(correlation_result_list)
        # dt.columns = ['filename', 'euc1_acc', 'euc2_acc', 'mah1_acc', 'mah2_acc', 'mah1_auc', 'mah2_auc', 'mah1_recall', 'mah2_recall',
        #               'q_acc', 'lo_acc', 'dis_acc', 'df_acc', 'q_auc', 'lo_auc', 'dis_auc', 'df_auc', 'q_recall', 'lo_recall', 'dis_recall', 'df_recall']
        # dt.to_csv('C:/Users/lvaid/skku/Technometrics/Projects/Deep Stacking Network/experiment/binary_correlation_result_again2.csv',
        #            index= None)
        # print("\n")