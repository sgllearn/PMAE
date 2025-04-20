import os
import logging
from datetime import datetime
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix


class Exp_Conf(object):
    def __init__(self, PY_FILE, SUPER_PARAM, sys_argv, dataSet='OSINT'):
        self.SUPER_PARAM = SUPER_PARAM
        self.sys_argv = sys_argv
        self.LOAD_WEIGHTS_FLAG = False
        if len(self.sys_argv) >= 2 and 'load_weights=1' in self.sys_argv[1]:  # 默认不产生时间戳前缀，除非传入time参数
            self.LOAD_WEIGHTS_FLAG = True
        self.TIME_PREFIX = ''
        if len(self.sys_argv) >= 2 and 'time_prefix=1' in self.sys_argv[1]:  # 默认不产生时间戳前缀，除非传入time参数
            self.TIME_PREFIX = datetime.now().strftime("%Y%m%d%H%M") + '_'
        self.PY_FILE = PY_FILE
        self.FILE_PREFIX = self.TIME_PREFIX + self.PY_FILE + '_'+ self.SUPER_PARAM
        self.TRAIN_LOG_FILE = 'LOG/' + self.FILE_PREFIX + '_training.log' + '.txt'
        self.MODEL_FILE = os.path.join('MODEL', self.FILE_PREFIX + '_model_weights' + '.h5')
        self.FMODEL_FILE = os.path.join('FMODEL', self.FILE_PREFIX + '_fmodel_weights' + '.h5')
        self.CONFUSION_MATRIX_FILE = 'LOG/' + self.FILE_PREFIX + '_Confusion_Matrix' + '.png'
        self.CONFUSION_MATRIX_LOG_FILE = 'LOG/' + self.FILE_PREFIX + '_Confusion_Matrix_' + '.txt'
        # 由低到高排序
        self.label_order = [
            'w32.virut', 'cryptowall', 'bedep', 'hesperbot', 'tempedre', 'beebone',
            'volatile', 'bamital', 'proslikefan', 'corebot', 'fobber', 'padcrypt', 'ramdo',
            'matsnu', 'nymaim', 'geodo', 'dircrypt', 'cryptolocker', 'pushdo', 'locky',
            'dyre', 'pykspa', 'qadars', 'suppobox', 'shifu', 'symmi', 'kraken',
            'shiotob-urlzone-bebloh', 'qakbot', 'ranbyus', 'simda', 'murofet', 'tinba',
            'necurs', 'ramnit', 'post', 'banjori'
        ]
        if dataSet == '360':
            self.label_order = ['tofsee', 'gspy', 'proslikefan', 'vidro', 'bamital', 'padcrypt', 'pykspa_v2_real', 'tempedreve', 'vawtrak', 'fobber_v1', 'fobber_v2', 'nymaim', 'dircrypt', 'conficker', 'pykspa_v2_fake', 'matsnu', 'chinad', 'dyre', 'cryptolocker', 'locky', 'qadars', 'suppobox', 'shifu', 'symmi', 'ranbyus', 'murofet', 'necurs', 'virut', 'gameover', 'ramnit', 'simda', 'pykspa_v1', 'tinba', 'rovnix', 'emotet', 'banjori']

        # 配置日志记录
        logging.basicConfig(filename=self.TRAIN_LOG_FILE, level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
    def log(self, info):
        logging.info(info)
    def check_load_weights(self, fmodel_flag=False):
        if fmodel_flag:
            self.log("Check FEX load weights.")
            return self.LOAD_WEIGHTS_FLAG and os.path.exists(self.FMODEL_FILE)
        else:
            self.log("Check load weights.")
            return self.LOAD_WEIGHTS_FLAG and os.path.exists(self.MODEL_FILE)
    def load_weights(self, model, fmodel_flag=False):
        if fmodel_flag:
            self.log("Loading existing FEX model weights.")
            model.load_weights(self.FMODEL_FILE)
        else:
            self.log("Loading existing model weights.")
            model.load_weights(self.MODEL_FILE)
    def save_weights(self, model, fmodel_flag=False):
        if fmodel_flag:
            self.log("Saving FEX model weights.")
            model_dir = os.path.dirname(self.FMODEL_FILE)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_weights(self.FMODEL_FILE)
        else:
            self.log("Saving model weights.")
            model_dir = os.path.dirname(self.MODEL_FILE)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model.save_weights(self.MODEL_FILE)
    def log_train(self, history):
        metrics = history.history.keys()
        num_epochs = len(history.epoch)
        for epoch in range(1, num_epochs + 1):
            epoch_result = []
            for metric in metrics:
                value = history.history[metric][epoch - 1]  # 列表索引从0开始，但轮数从1开始
                epoch_result.append(f"{metric}={value:.4f}")  # 格式化值到小数点后4位
            # 打印这一轮的结果，将指标和值用逗号分隔并连接成一个字符串
            logging.info(f"Epoch {epoch}/{num_epochs}:" + ", ".join(epoch_result))
        # 记录训练结束
        logging.info("Model training completed.")
    def log_evaluate(self, loss, accuracy):
        self.log(f"Test Loss: {loss}")
        self.log(f"Test Accuracy: {accuracy}")
    def log_test(self, y_true_classes, y_pred_classes, target_names):
        # 打印分类报告
        report = classification_report(y_true_classes, y_pred_classes, target_names=target_names, digits=4)
        logging.info('\r\n' + report)
        accuracy = accuracy_score(y_true_classes, y_pred_classes)
        logging.info(f"Accuracy: {accuracy}")



        # 生成混淆矩阵
        conf_matrix = confusion_matrix(y_true_classes, y_pred_classes)


        # 获取当前标签顺序
        current_label_order = target_names
        # 创建一个从当前标签到自定义标签顺序的映射
        label_mapping = {label: idx for idx, label in enumerate(current_label_order)}
        custom_label_indices = [label_mapping[label] for label in self.label_order]

        # 重新排序混淆矩阵
        conf_matrix_reordered = conf_matrix[:, custom_label_indices][custom_label_indices, :]
        conf_matrix_norm_reordered = conf_matrix_reordered.astype('float') / conf_matrix_reordered.sum(axis=1)[:,
                                                                             np.newaxis]
        # 保存混淆矩阵日志结果
        np.savetxt(self.CONFUSION_MATRIX_LOG_FILE, conf_matrix_reordered, delimiter='\t', fmt='%.4f')
        # 生成热力图
        plt.figure(figsize=(16, 14))
        sns.heatmap(conf_matrix_norm_reordered, annot=True, fmt='.2f', cmap='Blues', xticklabels=self.label_order,
                    yticklabels=self.label_order)
        plt.subplots_adjust(left=0.1, right=0.95, top=0.9, bottom=0.1)  # 调整图表边距
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Normalized Confusion Matrix (normalized)')
        # 保存图片到文件
        plt.savefig(self.CONFUSION_MATRIX_FILE)
        # plt.show()
