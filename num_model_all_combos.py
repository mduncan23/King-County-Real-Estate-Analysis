import statsmodels.api as sm
import pandas as pd
from itertools import combinations
from sklearn.metrics import mean_absolute_error, mean_squared_error


def simple_num_model_all_combos(df, target_var):
    
    '''
    input: dataframe of numerical features and this will run a linear regression 
    model for all columns against the input target var.
    
    return: r^2, adjusted r^2, f-stat prob, const coef, predictor coef, const p-val and predictor p-val.
    return values will be sorted in descending order based on adjusted r^2 values.
    
    Unlike usual models, target_var should be included in the same dataframe as potential x_vars.
    
    df should be a dataframe.
    target_var should be a column name in a string form.
    
    example: simple_num_model_all_combos(df_numeric, 'price')
    '''
    column = []
    r2 = []
    r2_adj = []
    f_stat_p_val = []
    const_coefs = []
    predictor_coef = []
    const_p_val = []
    predictor_p_val =[]
    
    
    for col in df:
        if col != target_var:
            X = df[[col]]
            y = df[target_var]

            model = sm.OLS(y, sm.add_constant(X)).fit()
            column.append(col)
            r2.append(model.rsquared)
            r2_adj.append(model.rsquared_adj)
            f_stat_p_val.append(model.f_pvalue)
            const_coefs.append(model.params.values[0])
            const_p_val.append(model.pvalues.values[0])
            predictor_coef.append(model.params.values[1])
            predictor_p_val.append(model.pvalues.values[1])
        else:
            continue
  
        
    col_list = [column, r2, r2_adj, f_stat_p_val, const_coefs, const_p_val, predictor_coef, predictor_p_val]
    col_list_names = ['column', 'r2', 'r2_adj', 'f_stat_p_val', 'const_coefs', 'const_p_val', 'predictor_coef', 'predictor_p_val']

    output_df = pd.DataFrame(col_list, index = col_list_names)

    output_df_transposed = output_df.T
            
    return output_df_transposed.sort_values('r2_adj', ascending=False)













def multi_num_model_all_combos(df, target_var):
    
    '''
    input: dataframe of numerical features and this will run all variations 
    of a linear regression model for each combination of features against the input target var.
    
    return: r^2, adjusted r^2, f-stat prob
    return values will be sorted in descending order based on adjusted r^2 values.
    
    Unlike usual models, target_var should be included in the same dataframe as potential x_vars.
    
    df should be a dataframe.
    target_var should be a column name in a string form.
    
    example: multi_num_model_all_combos(df_numeric, 'price')
    '''
    
    column = []
    r2 = []
    r2_adj = []
    f_stat_p_val = []
    const_coefs = []
    const_p_val = []

#     predictor_coef = []
#     predictor_p_val =[]
    
    list_of_combos = []
    p_value_good = []
    MAE = []
    RMSE = []
#     target_var = 'price'

    function_df = df.drop([target_var], axis=1).copy()
    for i, x in enumerate(function_df):
        list_of_combos.append(list(combinations(function_df, i+1)))
    
    for i in range(len(list_of_combos)):

        for combo in list_of_combos[i]:
            # created a temp list to hold each column name combination
            temp_list = []
            for x in combo:
                temp_list.append(x)
                X = function_df[temp_list]
                y = df[target_var]

            model = sm.OLS(y, sm.add_constant(X)).fit()
            column.append(temp_list)
            r2.append(model.rsquared)
            r2_adj.append(model.rsquared_adj)
            f_stat_p_val.append(model.f_pvalue)
            const_coefs.append(round(model.params.values[0], 4))
            const_p_val.append(round(model.pvalues.values[0], 4))
            good = 0
            for p in model.pvalues:
                total = len(model.pvalues)
                if p <= .05:
                    good += 1
            p_value_good.append((good/total)*100)
            MAE.append(mean_absolute_error(y, model.predict(sm.add_constant(X))))
            RMSE.append(mean_squared_error(y, model.predict(sm.add_constant(X)), squared=False))


#             if i == range(len(list_of_combos))[-1]:
#             for x in list_of_combos[i]:
#                 p_value_names.append(x)
                   
#             predictor_coef.append(model.params.values[1])
#             predictor_p_val.append(model.pvalues.values[1])
  
        
    col_list = [column, r2, r2_adj, f_stat_p_val, const_coefs, const_p_val, p_value_good, MAE, RMSE]
#                 , predictor_coef, predictor_p_val]
    col_list_names = ['column', 'r2', 'r2_adj', 'f_stat_p_val', 'const_coefs', 'const_p_val', '%p_val < .05', 'MAE', 'RMSE']
#                       , 'predictor_coef', 'predictor_p_val']

    output_df = pd.DataFrame(col_list, index = col_list_names)

    output_df_transposed = output_df.T
    
    output_df_transposed = output_df_transposed.sort_values('r2_adj', ascending=False).reset_index()
    
    
    top_3 = f"The top three combos are: \n\
    - {output_df_transposed['column'][0]} \n\
    \t- adj_r2: {output_df_transposed['r2_adj'][0]} \n\
    \t- ratio of p-vals <.05: {output_df_transposed['%p_val < .05'][0]}\n\
    - {output_df_transposed['column'][1]} \n\
    \t- adj_r2: {output_df_transposed['r2_adj'][1]}, \n\
    \t- ratio of p-vals <.05: {output_df_transposed['%p_val < .05'][1]}\n\
    - {output_df_transposed['column'][2]} \n\
    \t- adj_r2: {output_df_transposed['r2_adj'][2]}, \n\
    \t- ratio of p-vals <.05: {output_df_transposed['%p_val < .05'][2]}"
    
    # trying to get p_values for all coefs into the df but this is hard and i'm gonna comment this out and stop
#     for x in p_value_names:
#         for name in x:
#             output_df_transposed[f"p_val {name}"] = np.nan
            
    return output_df_transposed, print(top_3)









def base_check_for_category(df, category_column):
    
    '''
    Input a dataframe and specify a target column from the dataframe. 
    The dataframe needs to have categorical information only.
    
    Output is a stripplot of the category, statistics on the avg home price 
    when grouped by the category column, and a one-hot encoded model with the 
    first col dropped.
    '''
    
# Check how the categories in the feature compare to the price distribution
    stats = df.groupby([category_column])['price'].describe()
    
# Creating a chart of the column categories    
#     temp_list = list(df[category_column].unique())
#     sorted_list = sorted(temp_list)
    test_df = df[category_column].sort_values()
    
    mean_list = []
    for i in df.groupby([category_column])['price'].mean().sort_values().index:
        mean_list.append(i)

    fig, ax = plt.subplots(figsize=(15,10))
    sns.stripplot(x=test_df, y=df['price'], order=mean_list, palette="tab10")

    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    colors = colors *2
    
    for i, category in enumerate(mean_list):
#         z = i+1
        ax.axhline(y=df.groupby([category_column])['price'].mean().sort_values().values[i],
                   color = colors[i], label = f"{category}");
    ax.legend()

# Model the category
    y = df['price']
    X_cat = df[[category_column]].copy()
    X_cat['sqft_living'] = df_numeric['sqft_living'].copy()
    X_cat = pd.get_dummies(X_cat, columns=[category_column], drop_first=True)

    cat_results = sm.OLS(endog = y, exog = sm.add_constant(X_cat)).fit()
    
# get a summary of the model
    cat_summary = cat_results.summary()
    return stats, cat_results, cat_summary