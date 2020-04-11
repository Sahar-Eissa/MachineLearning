#!/usr/bin/env python
# coding: utf-8

# In[1]:


get_ipython().system("git clone https://github.com/fastai/course-v4 'drive/My Drive/course-v4'")


# In[2]:


get_ipython().system('pip install fastai2')


# In[3]:


from fastai2 import *
get_ipython().system('pip install graphviz ipywidgets matplotlib nbdev>=0.2.12 pandas scikit_learn ')


# In[4]:


from nbdev.showdoc import *
from ipywidgets import widgets
from pandas.api.types import CategoricalDtype


# In[5]:


get_ipython().system('pip install azure-cognitiveservices-search-imagesearch')


# In[6]:


from azure.cognitiveservices.search.imagesearch import ImageSearchClient as api
from msrest.authentication import CognitiveServicesCredentials as auth


# In[7]:


from fastai2.vision.all import *
get_ipython().system('pip install utils')
from utils import *
from fastai2.vision.widgets import *


# In[8]:


key = '173e7aa99dc34bdeabacd319ac5c4a92'


# In[9]:


def search_images_bing(key, term, min_sz=128):
    client = api('https://api.cognitive.microsoft.com', auth(key))
    return L(client.images.search(query=term, count=150, min_height=min_sz, min_width=min_sz).value)


# In[10]:


results = search_images_bing(key, 'girls kids')
ims = results.attrgot('content_url')
len(ims)


# In[11]:


ims[0]


# In[12]:


kids= 'girls', 'boys'


# In[13]:


path = Path('media')
if not path.exists():
    path.mkdir()
    for o in kids:
        dest = (path/o)
        dest.mkdir(exist_ok=True)
        results = search_images_bing(key, f'{o} kids')
        download_images(dest, urls=results.attrgot('content_url'))


# In[14]:


fns = get_image_files(path)
fns


# In[15]:


fns[0]


# In[16]:


to_see= 'media/girls/00000000.jpg'
download_url(fns[0], to_see)
im = Image.open(to_see)
im.to_thumb(128,128)


# In[23]:


get_ipython().system('pip install Pillow')
from PIL import Image 


# In[24]:


get_ipython().system('pip install webp')


# In[25]:


failed = verify_images(fns)
failed


# In[26]:


failed.map(Path.unlink);


# In[28]:


media_data = DataBlock(
    blocks = (ImageBlock, CategoryBlock),
    get_items=get_image_files,
    splitter=RandomSplitter(valid_pct=0.3, seed=42),
    get_y=parent_label,
    item_tfms=Resize(128))


# In[29]:


dls= media_data.dataloaders(path)


# In[30]:


dls.valid.show_batch(max_n=4, nrows=1)


# In[31]:


media_data = media_data.new(
    item_tfms=RandomResizedCrop(224, min_scale=0.5),
    batch_tfms=aug_transforms())
dls = media_data.dataloaders(path)
dls.valid.show_batch(max_n=4, nrows=1)


# In[32]:


learn= cnn_learner(dls, resnet18, metrics=error_rate)
learn.fine_tune(4)


# In[33]:


interp = ClassificationInterpretation.from_learner(learn)
interp.plot_confusion_matrix()


# In[34]:


interp.plot_top_losses(5, nrows=1)


# In[35]:


cleaner = ImageClassifierCleaner(learn)
cleaner


# In[39]:


for idx in cleaner.delete(): cleaner.fns[idx].unlink()


# In[40]:


learn.export()


# In[41]:


path = Path()
path.ls(file_exts='.pkl')


# In[42]:


learn_inf = load_learner(path/'export.pkl', cpu=True)


# In[43]:


learn_inf.predict('media/girls/00000000.jpg')


# In[44]:


learn_inf.dls.vocab


# In[46]:


get_ipython().system('pip install ipywidgets')


# In[47]:


pip install Voila


# In[140]:



from IPython.display import display
btn_upload = widgets.FileUpload()
btn_upload






# In[141]:


img = PILImage.create(btn_upload.data[-1])


# In[142]:


out_pl = widgets.Output()
out_pl.clear_output()
with out_pl: display(img.to_thumb(128,128))
out_pl


# In[143]:


pred,pred_idx,probs = learn_inf.predict(img)


# In[144]:


lbl_pred = widgets.Label()
lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'
lbl_pred


# In[145]:


btn_run = widgets.Button(description='Classify')
btn_run


# In[146]:


def on_click_classify(change):
    img = PILImage.create(btn_upload.data[-1])
    out_pl.clear_output()
    with out_pl: display(img.to_thumb(128,128))
    pred,pred_idx,probs = learn_inf.predict(img)
    lbl_pred.value = f'Prediction: {pred}; Probability: {probs[pred_idx]:.04f}'

btn_run.on_click(on_click_classify)


# In[147]:


btn_upload = widgets.FileUpload()


# In[148]:


VBox([widgets.Label('Diversity'), 
      btn_upload, btn_run, out_pl, lbl_pred])


# In[149]:


get_ipython().system('pip install voila')
get_ipython().system('jupyter serverextension enable voila --sys-prefix')


# In[ ]:





# In[ ]:




