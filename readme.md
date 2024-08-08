# llamaindex+Internlm2 RAG实践

## 笔记

此文档可能步骤较快，有任何不明白的可直接私聊我或者给我留言，谢谢~

### 先简单了解下什么是RAG？

RAG（Retrieval-Augmented Generation）技术是一种结合检索和生成功能的自然语言处理（NLP）技术。它通过从大型外部数据库中检索与输入问题相关的信息，来辅助生成模型回答问题，这种技术主要用于自然语言处理领域，尤其是在问答系统、对话系统和内容生成等应用中非常有效。RAG技术通过检索相关信息来增强语言模型的生成能力，使得生成的文本更加丰富、准确和相关。RAG技术的核心思想是将传统的检索技术与现代的自然语言生成技术相结合，以提高文本生成的准确性和相关性。

![](.\image-lx\1.png)

RAG模型的基本原理是将检索和生成两种技术结合起来，使模型能够在生成文本之前访问并利用大量外部信息。检索组件负责从一个大型的知识库中检索出与给定输入相关的信息，这个知识库可以是维基百科、专业期刊、书籍等任何形式的文档集合。生成组件则是一个预训练的Transformer模型（如GPT或BERT），它结合了原始输入和检索组件提供的外部信息来生成文本。

给模型注入新知识的方式，可以简单分为两种方式，一种是内部的，即更新模型的权重，另一个就是外部的方式，给模型注入格外的上下文或者说外部信息，不改变它的的权重。

总结下来就是一种需要重新训练更新模型的权重，另一种就是通过外部注入然后检索的形式不需要更新模型权重就能掌握新领域的知识。

**特点：**

不需要重新训练就可以获得新的信息感知，提高了模型内容回复的准确性，解决了内容实时更新性问题并且有效的缓解了幻觉问题。

## LlamaIndex HuggingFaceLLM

这里新建一个开发机选择30%A100*1的资源，镜像使用cuda11.7-conda操作，创建后我们进入开发机新建一个conda虚拟环境，名字随便取，我这里也是跟着教程来的。

```python
conda create -n llamaindex python=3.10
conda activate llamaindex
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
pip install einops
pip install  protobuf
#  安装 Llamaindex
conda activate llamaindex
pip install llama-index==0.10.38 llama-index-llms-huggingface==0.2.0 "transformers[torch]==4.41.1" "huggingface_hub[inference]==0.23.1" huggingface_hub==0.23.1 sentence-transformers==2.7.0 sentencepiece==0.2.0
```

接下来我就直接命令一步一步执行啦

```python
cd ~
mkdir llamaindex_demo model
cd ~/llamaindex_demo
touch download_hf.py
```



download_hf.py内容：

```python
import os

# 设置环境变量
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

# 下载模型
os.system('huggingface-cli download --resume-download sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 --local-dir /root/model/sentence-transformer')
```



继续执行后面的命令（我就不一一解释了，有任何不懂的可以百度，是实在不行直接私我）

```python
cd /root/llamaindex_demo
python download_hf.py
# 下载NTK资源
cd /root
git clone https://gitee.com/yzy0612/nltk_data.git  --branch gh-pages
cd nltk_data
mv packages/*  ./
cd tokenizers
unzip punkt.zip
cd ../taggers
unzip averaged_perceptron_tagger.zip
# 创建软链接（节省资源）
cd ~/llamaindex_demo
touch llamaindex_internlm.py
cd ~/llamaindex_demo
touch llamaindex_internlm.py
```

llamaindex_internlm.py代码：

```python
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.core.llms import ChatMessage
llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)

rsp = llm.chat(messages=[ChatMessage(content="xtuner是什么？")])
print(rsp)
```

继续执行：

```python
conda activate llamaindex
cd ~/llamaindex_demo/
python llamaindex_internlm.py
```

![](.\image-lx\4.png)

我们发现internlm2-chat-1_8b模型并不知道xtuner具体是什么，回答也是基于已经训练的知识库的内容去描述，并没有达到我们想要的效果，接下来就结合RAG的方式来查看效果。

## LlamaIndex RAG

安装LlamaIndex词向量的依赖，执行以下命令：

```python
conda activate llamaindex
pip install llama-index-embeddings-huggingface llama-index-embeddings-instructor
cd ~/llamaindex_demo
# 获取知识库
mkdir data
cd data
git clone https://github.com/InternLM/xtuner.git
mv xtuner/README_zh-CN.md ./
cd ~/llamaindex_demo
touch llamaindex_RAG.py
```



llamaindex_RAG.py内容如下：


```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

#初始化一个HuggingFaceEmbedding对象，用于将文本转换为向量表示
embed_model = HuggingFaceEmbedding(
#指定了一个预训练的sentence-transformer模型的路径
    model_name="/root/model/sentence-transformer"
)
#将创建的嵌入模型赋值给全局设置的embed_model属性，
#这样在后续的索引构建过程中就会使用这个模型。
Settings.embed_model = embed_model

llm = HuggingFaceLLM(
    model_name="/root/model/internlm2-chat-1_8b",
    tokenizer_name="/root/model/internlm2-chat-1_8b",
    model_kwargs={"trust_remote_code":True},
    tokenizer_kwargs={"trust_remote_code":True}
)
#设置全局的llm属性，这样在索引查询时会使用这个模型。
Settings.llm = llm

#从指定目录读取所有文档，并加载数据到内存中
documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
#创建一个VectorStoreIndex，并使用之前加载的文档来构建索引。
# 此索引将文档转换为向量，并存储这些向量以便于快速检索。
index = VectorStoreIndex.from_documents(documents)
# 创建一个查询引擎，这个引擎可以接收查询并返回相关文档的响应。
query_engine = index.as_query_engine()
response = query_engine.query("xtuner是什么?")

print(response)
```



接下来执行运行这个脚本即可：

```python
conda activate llamaindex
cd ~/llamaindex_demo/
python llamaindex_RAG.py
```



运行结果如下：

![](.\image-lx\5.png)

##  LlamaIndex web

运行一下命令：

```python
pip install streamlit==1.36.0
cd ~/llamaindex_demo
touch app.py
```



在`app.py`添加如下内容：

```python
import streamlit as st
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.huggingface import HuggingFaceLLM

st.set_page_config(page_title="llama_index_demo", page_icon="🦜🔗")
st.title("llama_index_demo")

# 初始化模型
@st.cache_resource
def init_models():
    embed_model = HuggingFaceEmbedding(
        model_name="/root/model/sentence-transformer"
    )
    Settings.embed_model = embed_model

    llm = HuggingFaceLLM(
        model_name="/root/model/internlm2-chat-1_8b",
        tokenizer_name="/root/model/internlm2-chat-1_8b",
        model_kwargs={"trust_remote_code": True},
        tokenizer_kwargs={"trust_remote_code": True}
    )
    Settings.llm = llm

    documents = SimpleDirectoryReader("/root/llamaindex_demo/data").load_data()
    index = VectorStoreIndex.from_documents(documents)
    query_engine = index.as_query_engine()

    return query_engine

# 检查是否需要初始化模型
if 'query_engine' not in st.session_state:
    st.session_state['query_engine'] = init_models()

def greet2(question):
    response = st.session_state['query_engine'].query(question)
    return response

      
# Store LLM generated responses
if "messages" not in st.session_state.keys():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]    

    # Display or clear chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])

def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "你好，我是你的助手，有什么我可以帮助你的吗？"}]

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

# Function for generating LLaMA2 response
def generate_llama_index_response(prompt_input):
    return greet2(prompt_input)

# User-provided prompt
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

# Gegenerate_llama_index_response last message is not from assistant
if st.session_state.messages[-1]["role"] != "assistant":
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = generate_llama_index_response(prompt)
            placeholder = st.empty()
            placeholder.markdown(response)
    message = {"role": "assistant", "content": response}
    st.session_state.messages.append(message)
```



接下来我们使用streamlit框架快速构建一个web应用，然后本地win+r执行`ssh -CNg -L 8501:127.0.0.1:8501 root@ssh.intern-ai.org.cn -p 44415`，这里44415是我的开发机ssh端口号，请修改为你自己InternStudio上面SSH连接的端口号即可。如下图显示

![](.\image-lx\6.png)

## 总结

这节小内容主要还是实现RAG技术帮助模型有效的缓解幻觉，采用的依旧是`internlm2-chat-1_8b`这个base模型，比较适合刚开始的熟悉RAG，但是这个base model还是有点基础，除了我们熟悉的代码理解外，还是使用了LLama_index库中的 `HuggingFaceLLM` 类和base模型交互，并且允许模型加载和执行远程代码，在RAG代码中，还使用了llama_index构建一个文本搜索引擎，将我们查询的文本（”Xtuner是什么“）转换成向量并且检索相关文档，采用的是sentence-transformer转换成的向量表示，基于句子来嵌入的，使用 `SimpleDirectoryReader` 从指定目录（`/root/llamaindex_demo/data`）读取所有文档，并加载数据到内存中，最后创建索引、创建查询引擎完成RAG的实现，我觉得输出有点慢，可能是gpu受限吧。
