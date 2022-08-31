import numpy as np
from sklearn import svm
from utils.logger import get_temp_logger


def train_boundary(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.02,
                   split_ratio=0.7,
                   invalid_value=None,
                   logger=None):
    """
      给出codes和scores的数组，训练一个线性SVM，最后返回相应的法向量
      Basically, the samples with highest attribute scores are treated as positive samples,
      while those with lowest scores as negative.
      目前latent—code只能接受1维的数据，也就是WP空间无法进行训练

      NOTE: The returned boundary is with shape (1, latent_space_dim), and also
      normalized with unit norm.

      Args:
        latent_codes: Input latent codes as training data.
        scores: Input attribute scores used to generate training labels.
        chosen_num_or_ratio: 决定选出多少条数据进行训练和验证，positive和negative都是chosen_num个
            如果在（0，1）之间，则选取chosen_num_or_ratio * latent_codes_num个
            如果是整数，则选取chosen_num_or_ratio个，
            最终和整体数量的一半进行比较（最多positive和negative每个一半），选出min(chosen_num, 0.5 * latent_codes_num)
        split_ratio: 训练集和验证集的划分
        invalid_value: 不考虑的值
        logger: logger，如果没有指定，则默认生成一个输入信息到屏幕的logger

      Returns:
        A decision boundary with type `numpy.ndarray`.

      Raises:
        ValueError: If the input `latent_codes` or `scores` are with invalid format.
      """

    if not logger:
         logger = get_temp_logger("train_boundary")
         # logger = setup_logger(work_dir='', logger_name='train_boundary')

    if (not isinstance(latent_codes, np.ndarray) or
            not len(latent_codes.shape) == 2):
        raise ValueError(f'Input `latent_codes` should be with type'
                         f'`numpy.ndarray`, and shape [num_samples, '
                         f'latent_space_dim]!')

    num_samples = latent_codes.shape[0]
    latent_space_dim = latent_codes.shape[1]

    if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
            not scores.shape[0] == num_samples or not scores.shape[1] == 1):
        raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                         f'shape [num_samples, 1], where `num_samples` should be '
                         f'exactly same as that of input `latent_codes`!')

    if chosen_num_or_ratio <= 0:
        raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                         f'but {chosen_num_or_ratio} received!')

    # 过滤掉等于invalid value的score
    logger.info(f'Filtering training data.')
    if invalid_value is not None:
        latent_codes = latent_codes[scores[:, 0] != invalid_value]
        scores = scores[scores[:, 0] != invalid_value]

    logger.info(f'Sorting scores to get positive and negative samples.')
    # 从小到大排序 ----》 逆序取出，从大到小排序
    sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
    latent_codes = latent_codes[sorted_idx]
    scores = scores[sorted_idx]
    num_samples = latent_codes.shape[0]
    if 0 < chosen_num_or_ratio <= 1:
        chosen_num = int(num_samples * chosen_num_or_ratio)
    else:
        chosen_num = int(chosen_num_or_ratio)
    chosen_num = min(chosen_num, num_samples // 2)

    # 划分数据集 chosen_num中划分出train-set和val-set
    # 未被选择的为remain-set
    # positive： 从高到低 train_num个
    # negative： 从低到高 train_num个
    logger.info(f'Spliting training and validation sets:')
    train_num = int(chosen_num * split_ratio)
    val_num = chosen_num - train_num

    # Positive samples.
    positive_idx = np.arange(chosen_num)
    np.random.shuffle(positive_idx)
    positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
    positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]

    # Negative samples.
    negative_idx = np.arange(chosen_num)
    np.random.shuffle(negative_idx)
    negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
    negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]

    # Training set.
    train_data = np.concatenate([positive_train, negative_train], axis=0)
    train_label = np.concatenate([np.ones(train_num, dtype=np.int),
                                  np.zeros(train_num, dtype=np.int)], axis=0)
    logger.info(f'  Training: {train_num} positive, {train_num} negative.')

    # Validation set.
    val_data = np.concatenate([positive_val, negative_val], axis=0)
    val_label = np.concatenate([np.ones(val_num, dtype=np.int),
                                np.zeros(val_num, dtype=np.int)], axis=0)
    logger.info(f'  Validation: {val_num} positive, {val_num} negative.')

    # Remaining set.
    remaining_num = num_samples - chosen_num * 2
    remaining_data = latent_codes[chosen_num:-chosen_num]
    remaining_scores = scores[chosen_num:-chosen_num]
    decision_value = (scores[0] + scores[-1]) / 2
    remaining_label = np.ones(remaining_num, dtype=np.int)
    remaining_label[remaining_scores.ravel() < decision_value] = 0
    remaining_positive_num = np.sum(remaining_label == 1)
    remaining_negative_num = np.sum(remaining_label == 0)
    logger.info(f'  Remaining: {remaining_positive_num} positive, '
                f'{remaining_negative_num} negative.')

    logger.info(f'Training boundary.')
    clf = svm.SVC(kernel='linear')
    classifier = clf.fit(train_data, train_label)
    logger.info(f'Finish training.')

    if val_num:
        val_prediction = classifier.predict(val_data)
        correct_num = np.sum(val_label == val_prediction)
        logger.info(f'Accuracy for validation set: '
                    f'{correct_num} / {val_num * 2} = '
                    f'{correct_num / (val_num * 2):.6f}')

    if remaining_num:
        remaining_prediction = classifier.predict(remaining_data)
        correct_num = np.sum(remaining_label == remaining_prediction)
        logger.info(f'Accuracy for remaining set: '
                    f'{correct_num} / {remaining_num} = '
                    f'{correct_num / remaining_num:.6f}')

    a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
    return a / np.linalg.norm(a)
