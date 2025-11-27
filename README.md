<h2><b>
    Overview of the project
</b></h2>

The project aims to classify correctly the quality of wines based on some of its chemical features. For that, I tested four different classification algorithms with slightly different characteristics: 
1. <b>Logistic Regression:</b> (# talk about some of its charracteristics); 
2. <b>Linear Discriminant Analysis:</b> which assumes that the variables are described by a gaussian distribution and has the same variance;
3. <b>Quadratic Discriminant Analysis:</b> which also assumes a gaussian distribution but not the same covariance matrix, what gives a quadratic decision boundary for the classification;
4. <b>Gaussian Naive Bayes:</b> (# talk about its characteristics).

Choosing the best performant algorithm, where the process is described on the following sections, I was able to achive ## % of accuracy, depending on the bussiness strategy, as described on the section ## (ref to the bussiness strategy section).

<h2><b>
    Connecting to SQL Server DataBase and Data Manipulation
</b></h2>

With the wine database mounted on SQL Server, we can use the <i>mysql</i> library to connect Python with it and get the chemichal features table:
```python
connection = connector.connect(
  host = '127.0.0.1',
  user = 'Leonardo-Loreti',
  password = '########',
  database = 'WineQT')

query = 'SELECT * FROM WineData'

wine = pd.read_sql(query, con = connection)
```

There were no <b>null</b> or <b>duplicated</b> data in the dataFrame, as was checked with the <i>.info()</i> and <i>.duplicated().sum()</i> commands, but the data was highly imbalanced as can be seen with the following histogram.

<p align = 'center'>
    <img src="Unbalanced dataset.png" height = '550' width = '550'
     alt = 'Histogram with the counts for the target variable (quality), with the original classes.'>
</p>

<p>In addition, some features showed a high <b>binary correlation</b>, which can be observed in the matrix correlation graph using <b>Pearson's correlation</b>, and some of them also showed a high <b>Variance Inflation Factor</b> (VIF), which indicates <b>multicollinearity</b> that can affect the accuracy of the coefficient estimates and degrade the inferential power of the models.</p>

<table align = 'center'>
<tr>
<th>VIF</th>
<th>Heatmap</th>
</tr>
<tr>
<td>
    - constant: 1.7108e6<br>
    - fixed acidity: 7.7845<br>
    - volatile acidity: 1.8799<br>
    - citric acid: 3.2245<br>
    - residual sugar: 1.7441<br>
    - chlorides: 1.5545<br>
    - free sulfur dioxide: 1.9075<br>
    - total sulfur dioxide: 2.1243<br>
    - density: 6.5979<br>
    - pH: 3.4034<br>
    - sulphates: 1.4955<br>
    - alcohol: 3.4108<br>
    - quality: 1.5981<br>
</td>
<td>
    <img src="Balanced dataset.png" height='550' width="550" />
</td>
</tr>
</table>

<p float="left">
    <ul>
        <li>constant: 1.7108E6</li>
        <li>fixed acidity: 7.7845</li>
        <li>volatile acidity: 1.8799</li>
        <li>citric acid: 3.2245</li>
        <li>residual sugar: 1.7441</li>
        <li>chlorides: 1.5545</li>
        <li>free sulfur dioxide: 1.9075</li>
        <li>total sulfur dioxide: 2.1243</li>
        <li>density: 6.5979</li>
        <li>pH: 3.4034</li>
        <li>sulphates: 1.4955</li>
        <li>alcohol: 3.4108</li>
        <li>quality: 1.5981</li>
    </ul>
    <img src="Balanced dataset.png" width="100" />
</p>


<div class="row">
    <div id='div1'>
        
    </div>
    <div id='div2'>
        <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/img/heatmap_originalFeatures.png" title="example image" class="img-fluid rounded z-depth-1" %}
    </div>
    </div>
</div>

<h4><b>
    Feature Creation
</b></h4>

To solve the problem of the high VIF and binary correlation between the variables fixed acidity, citric acid, density, alcohol, total sulful dioxide, it is possible to create new representative variables:
<ul>
    <li><b>total acidity</b> = fixe acidity + volatile acidity</li>
    <li><b>citric acid percentage</b> = citric acid/total acidity</li>
    <li><b>free sulfur dioxide percentage</b> = free sulfur dioxide/total sulful dioxide</li>
    <li><b>percentage of alcohol density</b> = alcohol/(100*density) (verificar se a divisão por 100 foi feita pq a densidade de alcool ficaria muito alta distoando da escala das outras variáveis)</li>
</ul>
