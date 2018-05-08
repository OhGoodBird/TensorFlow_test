
# coding: utf-8

# In[ ]:


from config import Config


# In[ ]:


config = Config()
print('wordemb_dim = %d' %(config.wordemb_dim))
print('label_dim = %d' %(config.label_dim))


# In[ ]:


from model import RNNModel
model = RNNModel(config)
model.build()
num_params = model.get_num_params()
print('num of params = %d' %(num_params))
model.train()

