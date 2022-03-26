import sklearn
import numpy as np
import prepare_data

from Feature import Feature
from Decoder import Decoder
from sklearn.model_selection import KFold

def train_avg(iterations, train_data, beam_size, cv_epoch_num):
    # data = prepare_data.read_file(train_file)
    data = train_data
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)
    n = 0
    for t in range(iterations):
        count = 0
        data_size = len(data)

        for line in data:
            n += 1
            y = line.split()
            z = decoder.beamSearch(line)
            if z != y:
                feature.update_avgWeight(y, z, n, t, data_size)

            train_seg = ' '.join(z)

            count += 1
            if count % 1000 == 0:
                print("iter %d , finish %.2f%%" % \
                        (t, (count / data_size) * 100))

        model_file = open('cv-' + str(cv_epoch_num) + "-model_result/model-" + \
                            str(t) + "_beam-size-" + str(beam_size) + '.pkl', 'wb')
        feature.save_model(model_file)
        model_file.close()
        print("segment with model-%d finish" % t)
        print("iteration %d finish" % t)

    feature.last_update(iterations, data_size)
    feature.cal_avg_weight(iterations, data_size)
    avg_model = open(
        'cv-' + str(cv_epoch_num) + "-model_result/avg-model_beam-size-" + str(beam_size) + '.pkl', 'wb')
    feature.save_model(avg_model)
    avg_model.close()
    print("segment with avg-model finish")


def test_avg(iterations, test_data, beam_size, cv_epoch_num):
    # data = prepare_data.read_file(test_file)
    data = test_data
    feature = Feature()
    decoder = Decoder(beam_size, feature.get_score)

    count = 0
    data_size = len(data)

    model_file = open(
        'cv-' + str(cv_epoch_num) + '-model_result/avg-model_beam-size-' + str(beam_size) + '.pkl', 'rb')
    feature.load_model(model_file)
    model_file.close()
    for line in data:
        z = decoder.beamSearch(line)
        seg_data = ' '.join(z)
        seg_data_file = 'cv-' + str(cv_epoch_num) + '-test_seg_data/avg-test-seg-data' + \
            '_beam-size-' + str(beam_size) + '.txt'
        with open(seg_data_file, 'a') as f:
            f.write(seg_data + '\n')
        count += 1
        if count % 1000 == 0:
            print("segment with avg-model, finish %.2f%%" %
                  ((count / data_size) * 100))
    f.close()
    print("segment with avg model finish")



if __name__ == '__main__':

    kf = KFold(n_splits=5, shuffle=True)

    train_file = 'data/fusion_data.txt'
    raw_data = prepare_data.read_file(train_file)
    cv_epoch_num = 1

    for train_id, test_id in kf.split(raw_data):
        print("TRAIN:", train_id, "TEST:", test_id)
        train_data, test_data = np.array(raw_data)[train_id], np.array(raw_data)[test_id]
        print("cross-valid epoch:", cv_epoch_num)

        train_avg(iterations=10, train_data=train_data, beam_size=16, cv_epoch_num=cv_epoch_num)
        test_avg(iterations=10, test_data=test_data, beam_size=16, cv_epoch_num=cv_epoch_num)

        cv_epoch_num += 1
        # print(len(raw_data), len(train_data), len(test_data))
        # break