## Website for Dongmin Kim

Available @ [carrtesy.github.io](https://carrtesy.github.io/)



## From Jupyter Notebook to Markdown
required: nbconvert < 6.0.0 (conda install nbconvert=5.6.1) 

```bash
jupyter nbconvert MNIST_Trials.ipynb --to markdown --template jekyll.tpl
```


## Building New Category

1. Make markdown file at _pages dir.
2. fix navigation.yml at _data dir.
