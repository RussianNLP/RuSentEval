echo 'Installing requirements...'

python -m pip install -r requirements.txt

echo 'Downloading fasttext model...'

mkdir models
wget -c http://files.deeppavlov.ai/embeddings/ft_native_300_ru_wiki_lenta_nltk_word_tokenize/ft_native_300_ru_wiki_lenta_nltk_word_tokenize.bin -P models
