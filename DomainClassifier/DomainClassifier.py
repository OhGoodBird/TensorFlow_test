
# coding: utf-8

# In[ ]:


from config import Config
from data_loader import DataLoader


# In[ ]:


config = Config()
print('wordemb_dim = %d' %(config.wordemb_dim))
print('label_dim = %d' %(config.label_dim))


# In[ ]:


from model import RNNModel
model = RNNModel(config)
print('loading train data ...')
train_dl = DataLoader(config.train_path, config.word2id, config.label2id, config.max_len, config.replaceDigit)
print('loading valid data ...')
valid_dl = DataLoader(config.valid_path, config.word2id, config.label2id, config.max_len, config.replaceDigit)
print('loading test data ...')
test_dl  = DataLoader(config.test_path, config.word2id, config.label2id, config.max_len, config.replaceDigit)

print('building model graph ...')
model.build()
num_params = model.get_num_params()
print('num of params = %d' %(num_params))
print('Start Training ...')
model.train(train_dl, valid_dl, test_dl)
print('Training Finish')

model.close_session()


print('Start Test')
import os
model.build()
model.restore_session(os.path.join(config.save_model_path, 'model.ckpt'))
test_acc, test_correct, test_num, test_loss = model.test(test_dl, 
                                                         batch_size=1024, 
                                                         output_file=os.path.join(config.output_result_path, 
                                                                         'predict.txt'))
print('Test  Accuracy = %.2f %% (%d / %d), loss = %.10f' %(test_acc, test_correct, test_num, test_loss))
print('Test Finish')


