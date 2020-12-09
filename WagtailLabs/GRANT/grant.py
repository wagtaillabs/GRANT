import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import _tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import check_random_state
import copy
import itertools 
import datetime
import gc

class grant:
  grafted_df = None 
  __version__ = "0.0.1"
  
  def __init__(self, rf, train_features, train_labels):
      self.rf = rf
      self.train_features = train_features
      self.train_labels = train_labels

  def graft(self, verbose=False):
    self.grafted_df = self.__graft(self.rf, self.train_features, verbose)

  def get_graft(self):
    if self.grafted_df is None: self.graft()
    return(self.grafted_df)

  def match_grafted(self, feature):
    if self.grafted_df is None: self.graft()
    lt_cols = self.grafted_df.columns[:-1:2]
    gt_cols = self.grafted_df.columns[1::2]
    col = lt_cols[0][1:]
    matches = [True] * len(self.grafted_df)
    for i in range(len(lt_cols)):
      col = lt_cols[i][1:]
      matches = matches & (self.grafted_df[lt_cols[i]] < feature[col].values[0]) & (feature[col].values[0] <= self.grafted_df[gt_cols[i]])
    return(self.grafted_df[matches])

  def match_amalgamated(self, feature):
    if self.amalgamate_df is None: raise Exception("Amalgamate has not been run.")
    lt_cols = self.amalgamate_df.columns[:-1:2]
    gt_cols = self.amalgamate_df.columns[1::2]
    col = lt_cols[0][1:]
    matches = [True] * len(self.amalgamate_df)
    for i in range(len(lt_cols)):
      col = lt_cols[i][1:]
      matches = matches & (self.amalgamate_df[lt_cols[i]] < feature[col].values[0]) & (feature[col].values[0] <= self.amalgamate_df[gt_cols[i]])
    return(self.amalgamate_df[matches])

  def reassemble(self, query='`value` == `value`'):
    if self.grafted_df is None: self.graft()
    return(self.__reassemble(self.grafted_df.query(query)))

  def amalgamate(self, threshold):
    if self.grafted_df is None: self.graft()
    self.amalgamate_df = self.__amalgamate(self.grafted_df, threshold).drop(['amalgamate_count', 'amalgamate_total', 'amalgamate_min', 'amalgamate_max', 
                                                                             'lead_diff', 'lag_diff', 'min_diff'], axis = 1)
    
  def get_amalgamated(self):
    if self.amalgamate_df is None: raise Exception("Amalgamate has not been run.")
    return(self.amalgamate_df)

  def reassemble_amalgamated(self, query='`value` == `value`'):
    if self.amalgamate_df is None: raise Exception("Amalgamate has not been run.")
    return(self.__reassemble(self.amalgamate_df.query(query)))

  def neighbour_sensitivity(self, feature):
    if self.grafted_df is None: self.graft()
    return(self.__neighbour_sensitivity(feature, self.grafted_df))

  def training_delta_trainer(self, trainer_feature, trainer_label):
    if self.grafted_df is None: self.graft()
    grafted_df = self.grafted_df.copy()
    grafted_df['trainer_delta'] = self.__training_delta_trainer(self.train_features, trainer_feature, trainer_label, grafted_df)
    return(grafted_df[grafted_df['trainer_delta'] != 0].sort_values(['trainer_delta']))

  def training_delta_trainee(self, trainee_feature, trainee_label):
    train_delta = self.train_features.copy()
    train_delta['trainee_delta'] = self.__training_delta_trainee(self.rf, train_delta, self.train_labels, trainee_feature, trainee_label)
    return(train_delta[train_delta['trainee_delta'] != 0].sort_values(['trainee_delta']))




  def __tree_to_dataframe(self, tree, feature_names, position):
      tree_ = tree.tree_
      feature_name = [
        feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
      ]

      data = [[val for pair in zip([-float('Inf') for feature in feature_names], 
                                  [float('Inf') for feature in feature_names]) for val in pair] + [0]]
      columns = [val for pair in zip(["<" + feature for feature in feature_names], 
                                    [feature + "<=" for feature in feature_names]) for val in pair] + ["value"]
      splits = pd.DataFrame(data, columns = columns).astype('float32')
      df = self.__tree_to_dataframe_recurse(splits, tree_, feature_name, 0, position)
      return(df)

  def __tree_to_dataframe_recurse(self, splits, tree_, feature_name, node, position):
    if tree_.feature[node] != _tree.TREE_UNDEFINED:
      splits_left = splits.copy()
      splits_right = splits.copy()
      name = feature_name[node]
      threshold = tree_.threshold[node]

      if threshold < threshold.astype('float32'):
        threshold = threshold - (threshold.astype('float32') - threshold)
      if threshold > threshold.astype('float32'):
        threshold = threshold + (threshold.astype('float32') - threshold)

      splits_left[name+"<="] = threshold.astype('float32')
      splits_left = self.__tree_to_dataframe_recurse(splits_left, tree_, feature_name, tree_.children_left[node], position)
      splits_right["<"+name] = threshold.astype('float32')
      splits_right = self.__tree_to_dataframe_recurse(splits_right, tree_, feature_name, tree_.children_right[node], position)

      splits = pd.concat([splits_left, splits_right])
    else:
      splits["value"] = tree_.value[node]
      splits["leaf"] = str(position) + "-" + str(node)
    return(splits)

  def __list_trees(self, rf, features):
    feature_names = features.columns
    merged_trees = self.__tree_to_dataframe(rf.estimators_[0], feature_names, 0)
    tree_list = [merged_trees]
    for i in range(1, len(rf.estimators_)):
      temp = self.__tree_to_dataframe(rf.estimators_[i], feature_names, i)
      merged_trees = merged_trees.append(temp, sort=False)
      tree_list[i:] = [temp]
    return(tree_list)

  def __score_leaves(self, in_rf, score_df):
    rf = copy.deepcopy(in_rf)
    rf.estimators_[0].tree_.value[:,0,0] = range(rf.estimators_[0].tree_.value.shape[0])
    leaves = np.expand_dims(rf.estimators_[0].predict(score_df).astype('int').astype('str'), 1)
    for i in range(1, len(rf.estimators_)):
      rf.estimators_[i].tree_.value[:,0,0] = range(rf.estimators_[i].tree_.value.shape[0])
      leaves = np.append(leaves, np.expand_dims(rf.estimators_[i].predict(score_df).astype('int').astype('str'), 1), axis=1)
    leaves=pd.DataFrame(data=leaves,
                        index=[i for i in range(leaves.shape[0])],
                        columns=['leaf'+str(i) for i in range(leaves.shape[1])])
    for i in range(len(rf.estimators_)):
      leaves['leaf'+str(i)] = str(i) + "-" + leaves['leaf'+str(i)]
    return(leaves)

  def __graft_leaves(self, tree_list, leaves):
    tree_df = tree_list[0]
    leaves['graft_id'] = range(len(leaves))
    df = leaves.merge(tree_df, left_on='leaf0', right_on='leaf').sort_values('graft_id')
    min_names = [col for col in df.columns if '=' in col]
    max_names = [col for col in df.columns if '<' in col and not '=' in col]
    avg_names = ['value']
    raw_columns = np.concatenate((max_names, min_names, leaves.columns), axis=0)
    ordered_columns = [val for pair in zip([feature for feature in max_names], 
                                          [feature for feature in min_names]) for val in pair]
    for i in range(1, len(tree_list)):
      tree_df = tree_list[i]
      temp = leaves.merge(tree_df, left_on='leaf'+str(i), right_on='leaf').sort_values('graft_id')
      min_values = np.concatenate((np.float32(np.expand_dims(df[min_names].to_numpy(), axis=2)), 
                                  np.float32(np.expand_dims(temp[min_names].to_numpy(), axis=2))), axis=2).min(axis=2)
      max_values = np.concatenate((np.float32(np.expand_dims(df[max_names].to_numpy(), axis=2)), 
                                  np.float32(np.expand_dims(temp[max_names].to_numpy(), axis=2))), axis=2).max(axis=2)
      avg_values = np.concatenate((np.expand_dims(df[avg_names].to_numpy(), axis=2), 
                                  np.expand_dims(temp[avg_names].to_numpy(), axis=2)), axis=2).sum(axis=2)
      data = np.concatenate((max_values, min_values, leaves), axis=1)
      df = pd.DataFrame(data=data,
                        index=df.index,
                        columns=raw_columns)
      df = df.set_index(['leaf0','leaf1'])
      df = df.drop('graft_id', axis=1)
      df = df[ordered_columns].astype(np.float32)
      df[avg_names[0]] = avg_values / len(tree_list)
    return(df)

  def __update_score_df(self, score_df_all, grafted_leaves):
    score_df = grafted_leaves[grafted_leaves.columns[1::2]].copy()
    score_df = score_df.rename(columns = lambda x : str(x)[:-2])
    score_df = np.nextafter(score_df, -1)
    
    score_next = score_df[0:0].copy()
    for c in range(score_df.columns.shape[0]):
      update = grafted_leaves.iloc[:,c*2]
      insert = score_df.copy()
      insert.iloc[:,c] = np.nextafter(update, -1)
      score_next = score_next.append(insert)
    score_next[score_next > 2**31-1] = np.float(2**31+1)
    score_next[score_next < -2**31+1] = np.float(-2**31+1)
    score_next = score_next.drop_duplicates()

    score_next = score_next.merge(score_df_all, how='left', indicator=True).copy()
    score_next = score_next[score_next['_merge']=='left_only'].drop('_merge', axis=1).copy()
    score_df_all = score_df_all.append(score_next)
    
    return([score_next, score_df_all])

  def __graft(self, rf, df, verbose):
    if verbose: print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    tree_list = self.__list_trees(rf, df)
    score_df = copy.deepcopy(df[0:1])
    score_df[:] = np.float(2**31-1)
    score_df = score_df.append(df)
    score_df_all = copy.deepcopy(score_df)
    leaves = self.__score_leaves(rf, score_df)
    leaves_out = leaves.copy()
    if verbose: print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + str(leaves_out.shape[0]) + ' - ' + str(score_df.shape[0]))
    grafted_leaves = self.__graft_leaves(tree_list, leaves)
    grafted_df = grafted_leaves.copy()
    while(True):
      [score_df, score_df_all] = self.__update_score_df(score_df_all, grafted_leaves)
      leaves = self.__score_leaves(rf, score_df)
      leaves = leaves.drop_duplicates()

      len_before = leaves_out.shape[0]
      leaves_out = leaves_out.append(leaves)
      leaves_out = leaves_out.drop_duplicates()
      if verbose: print (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S") + ' - ' + str(leaves_out.shape[0]) + ' - ' + str(score_df.shape[0]))
      grafted_leaves = self.__graft_leaves(tree_list, leaves)
      grafted_df = grafted_df.append(grafted_leaves)
      if(len_before == leaves_out.shape[0]):
        break
    grafted_df = grafted_df.drop_duplicates()
    return(grafted_df)

  def __split_graft(self, grafted_df):
    all_len = grafted_df.shape[0]
    lt_cols = grafted_df.columns[:-1:2]
    gt_cols = grafted_df.columns[1::2]
    most_even_split_diff = 999999999999999
    most_even_split_value = 0
    most_even_split_col = -1
    for i in range(len(lt_cols)):
      lt_col = lt_cols[i]
      gt_col = gt_cols[i]
      grafted_df = grafted_df.sort_values(gt_col)
      split_value = grafted_df[gt_col].values[int(grafted_df.shape[0] / 2)]
      split_value_inc = np.nextafter(split_value.astype('float32'), 1)
      lt_len = sum(grafted_df[lt_col] < split_value_inc)
      gt_len = sum(grafted_df[gt_col] > split_value_inc)
      while((lt_len == all_len)&(sum(grafted_df[gt_col].values < split_value) > 0)):
        split_value = max(grafted_df[gt_col].values[grafted_df[gt_col].values < split_value])
        split_value_inc = np.nextafter(split_value.astype('float32'), 1)
        lt_len = sum(grafted_df[lt_col] < split_value_inc)
        gt_len = sum(grafted_df[gt_col] > split_value_inc)
      while((gt_len == all_len)&(sum(grafted_df[gt_col].values > split_value) > 0)):
        split_value = min(grafted_df[gt_col].values[grafted_df[gt_col].values > split_value])
        split_value_inc = np.nextafter(split_value.astype('float32'), 1)
        lt_len = sum(grafted_df[lt_col] < split_value_inc)
        gt_len = sum(grafted_df[gt_col] > split_value_inc)
      split_diff = (abs(lt_len - gt_len)+2)**2 + ((lt_len + gt_len)**2)
      if (split_diff < most_even_split_diff) & (lt_len > 0) & (gt_len > 0) & (lt_len < all_len) & (gt_len < all_len):
        most_even_split_diff = split_diff
        most_even_split_value = split_value
        most_even_split_col = i
    return [gt_cols[most_even_split_col], lt_cols[most_even_split_col], most_even_split_value]

  def __dataframe_to_tree_recurse(self, grafted_df, node = 0):
    if grafted_df.shape[0] == 1:
      return([[_tree.TREE_LEAF], [_tree.TREE_LEAF], [_tree.TREE_UNDEFINED], [_tree.TREE_UNDEFINED], [grafted_df['value'].iloc[0]]])

    [most_even_split_gt_col, most_even_split_lt_col, most_even_split_value] = self.__split_graft(grafted_df)

    grafted_df_lt = grafted_df[grafted_df[most_even_split_lt_col] < most_even_split_value]
    grafted_df_gt = grafted_df[grafted_df[most_even_split_gt_col] > most_even_split_value]
    left = self.__dataframe_to_tree_recurse(grafted_df_lt, node+1)
    right = self.__dataframe_to_tree_recurse(grafted_df_gt, node + len(left[0]) + 1)

    children_left = [node + 1] + left[0] + right[0]
    children_right = [node + len(left[0]) + 1] + left[1] + right[1]
    feature = [most_even_split_gt_col] + left[2] + right[2]
    threshold = [most_even_split_value] + left[3] + right[3]
    value = [0] + left[4] + right[4]

    return([children_left, children_right, feature, threshold, value])

  def __reassemble(self, grafted_df):
    [children_left, children_right, feature, threshold, value] = self.__dataframe_to_tree_recurse(grafted_df)
    grafted_X = grafted_df[grafted_df.columns[1::2]].copy().rename(columns = lambda x : str(x)[:-2])
    grafted_X = grafted_X.append(grafted_X).append(grafted_X)
    grafted_Y = grafted_df['value']
    grafted_Y = grafted_Y.append(grafted_Y).append(grafted_Y)
    grafted_X[:] = np.expand_dims(range(len(grafted_X)), 1)
    grafted_Y[:] = range(len(grafted_Y))

    grafted_dt = DecisionTreeRegressor()
    grafted_dt = grafted_dt.fit(grafted_X, grafted_Y)
    grafted_dt.tree_.children_left[0:len(children_left)] = children_left
    grafted_dt.tree_.children_right[0:len(children_right)] = children_right
    grafted_dt.tree_.threshold[0:len(threshold)] = np.nextafter(threshold, 1)
    grafted_dt.tree_.value[0:len(value)] = np.expand_dims(np.expand_dims(value, 1), 1)
    
    threshold = np.float32(threshold)
    for i in range(len(threshold)):
      threshold[i] = threshold[i] + abs(threshold[i] - np.nextafter(threshold[i], 1))*10
    grafted_dt.tree_.threshold[0:len(threshold)] = threshold

    cols = grafted_df.columns[1::2].tolist()
    feature_indexes = [cols.index(x) if x != -2 else -2 for x in feature]
    grafted_dt.tree_.feature[0:len(feature_indexes)] = feature_indexes

    return(grafted_dt)

  def __find_neighbours(self, val_feature, grafted_df):
    val_grafted = self.match_grafted(val_feature)

    lt_cols = grafted_df.columns[:-1:2]
    gt_cols = grafted_df.columns[1::2]
    col = lt_cols[0]
    matches = (grafted_df[lt_cols[0]] <= val_grafted[gt_cols[0]].values[0]) & (val_grafted[lt_cols[0]].values[0] <= grafted_df[gt_cols[0]])
    for i in range(1, len(lt_cols)):
      col = lt_cols[i]
      matches = matches & (grafted_df[lt_cols[i]] <= val_grafted[gt_cols[i]].values[0]) & (val_grafted[lt_cols[i]].values[0] <= grafted_df[gt_cols[i]])
    return(grafted_df[matches])

  def __neighbour_sensitivity(self, feature, grafted_df):
    neighbours = self.__find_neighbours(feature, grafted_df)
    sensitivities = neighbours[['value']].copy()
    sensitivities.loc[:, 'value_delta'] = neighbours['value'].values - self.match_grafted(feature)['value'].values
    sensitivities = sensitivities.drop('value', axis=1)
    cols = feature.columns
    for i in range(len(cols)):
      col = cols[i]
      sensitivities.loc[:, col+'_delta'] = (neighbours['<'+col].values - feature[col].values[0]).clip(min=0) + (neighbours[col+'<='].values - feature[col].values[0]).clip(max=0)
    return(sensitivities)

  def __match_leaves(self, rf, train_df, train_leaves, graft, tree):
    n_samples = train_df.shape[0]
    random_instance = check_random_state(rf.estimators_[tree].random_state)
    sample_indices = random_instance.randint(0, n_samples, n_samples)
    indice_count = np.unique(np.append(list(range(n_samples)), sample_indices), return_counts=True)[1] - 1            #create a list of all indices, append the sampled indicies, count them, and then subtract one to remove the influence of the total list. Guarantees that unsampled records return a zero, which is important to make sure everything lines up later.
    leaf_matches = (train_leaves['leaf'+str(tree)] == graft['leaf'+str(tree)].tolist()[0]).astype(int) * indice_count #create a list of the number of times each training record that was used to train the leaf of interest was sampled
    return(leaf_matches)

  def __training_delta_trainer(self, train_features, trainer_feature, trainer_label, grafted_df):
    train_leaves = self.__score_leaves(rf, train_features)                                    #find the leaves for the training records
    trainer_leaf = self.__score_leaves(rf, trainer_feature)                                   #find the leaves for the answer records
    delta = [0] * len(grafted_df)                                                             #define a list of zeros that's as long as the graft set - our starting assumption is that the answer graft doesn't contribute to any grafts.
    for i in range(len(trainer_leaf.columns)):                                                #iterate through the columns of answer_leaf (i.e. the number of trees in the Random Forest)
      matched_leaves = self.__match_leaves(rf, train_features, train_leaves, trainer_leaf, i) #find which training records contributed for the answer record
      if np.sum(matched_leaves[trainer_feature.index[0] == matched_leaves.index]) > 0:        #test whether that includes the answer record (training records are randomly sampled for each tree)
        match = (grafted_df.reset_index()['leaf'+str(i)] == trainer_leaf['leaf'+str(i)][0])   #get a list of all graft records that use the leaf of interest
        overlap = list(np.asarray(match) * np.sum(matched_leaves[trainer_feature.index[0] == matched_leaves.index]) / np.sum(matched_leaves))  #calculate the percentage of contributing training records that is the answer record
        change = (np.asarray(trainer_label) - grafted_df['value'].tolist()) * overlap / len(trainer_leaf.columns)                              #calculate how the answer record changed the overall answer
        delta = [sum(x) for x in zip(delta, change)]                                          #add the change to the rolling delta
    return(delta)

  def __training_overlap(self, rf, train_df, trainee_feature):
    train_leaves = self.__score_leaves(rf, train_df)
    trainee_leaves = self.__score_leaves(rf, trainee_feature)

    leaf_matches = self.__match_leaves(rf, train_df, train_leaves, trainee_leaves, 0)
    total = 1 / np.sum(leaf_matches)
    matches = leaf_matches * total
    for i in range(1, len(rf.estimators_)):
      leaf_matches = self.__match_leaves(rf, train_df, train_leaves, trainee_leaves, i)
      matches = matches + leaf_matches * (1 / np.sum(leaf_matches))
      total = total + 1 / np.sum(leaf_matches)
    matches = matches / total
    return(matches)

  def __training_delta_trainee(self, rf, train_df, train_labels, trainee_feature, trainee_label):
    overlap = self.__training_overlap(rf, train_df, trainee_feature)
    train_df['training_overlap'] = overlap
    return((train_labels - trainee_label.tolist()[0]) * overlap / np.sum(overlap))

  def __initialise_shift(self, amalgamated_df, shifts):
    amalgamated_df = amalgamated_df.sort_values('value')
    amalgamated_df['lead_diff'] = amalgamated_df['value'].shift(-shifts) - amalgamated_df['value']
    amalgamated_df['lag_diff'] = amalgamated_df['value'] - amalgamated_df['value'].shift(shifts)
    amalgamated_df['min_diff'] = amalgamated_df[['lead_diff', 'lag_diff']].min(axis=1)
    amalgamated_df_shift = amalgamated_df.shift(shifts)
    return [amalgamated_df, amalgamated_df_shift]

  def __find_matches(self, amalgamated_df, amalgamated_df_shift, shifts):
    drops = ['value', 'lead_diff', 'lag_diff', 'min_diff', 'amalgamate_count', 'amalgamate_total', 'amalgamate_min', 'amalgamate_max']
    match_columns = len(amalgamated_df.drop(drops, axis=1).columns) - 2
    matches = np.sum(amalgamated_df.drop(drops, axis=1) == amalgamated_df_shift.drop(drops, axis=1), axis=1) == match_columns
    return(matches)

  def __find_close_shift_matched_box(self, amalgamated_df, amalgamated_df_shift, shifts, matches, row, threshold):
    grafted_row = amalgamated_df[matches][row:row+1]
    grafted_row_shift = amalgamated_df_shift[matches][row:row+1]
    for c_lt in grafted_row.columns[:-8:2]:   #-8:2 because we only need to consider the <x & x<= pairs
      c_gt = c_lt[1:]+'<='
      if (grafted_row[c_lt][0] == grafted_row_shift[c_gt][0]) | (grafted_row[c_gt][0] == grafted_row_shift[c_lt][0]):
        if ((abs(grafted_row['amalgamate_min'][0] - grafted_row_shift['amalgamate_max'][0]) < threshold) &
            (abs(grafted_row['amalgamate_max'][0] - grafted_row_shift['amalgamate_min'][0]) < threshold)):
          return([c_lt, c_gt])
    return([None, None])                      #return None if we don't find any usable matches

  def __combine_grafts(self, amalgamated_df, amalgamated_df_shift, shifts, matches, row, c_lt, c_gt):
    grafted_row = amalgamated_df[matches][row:row+1]
    grafted_row_shift = amalgamated_df_shift[matches][row:row+1]
    amalgamated_df.at[grafted_row.index, c_lt]=np.min(grafted_row[c_lt][0:1].append(grafted_row_shift[c_lt][0:1]))
    amalgamated_df.at[grafted_row.index, c_gt]=np.max(grafted_row[c_gt][0:1].append(grafted_row_shift[c_gt][0:1]))
    amalgamated_df = amalgamated_df.drop(amalgamated_df[:-shifts][matches[shifts:].to_list()][row:row+1].index)

    count = np.sum(grafted_row['amalgamate_count'][0:1].append(grafted_row_shift['amalgamate_count'][0:1]))
    total = np.sum(grafted_row['amalgamate_total'][0:1].append(grafted_row_shift['amalgamate_total'][0:1]))
    min = np.min(grafted_row['amalgamate_min'][0:1].append(grafted_row_shift['amalgamate_min'][0:1]))
    max = np.max(grafted_row['amalgamate_max'][0:1].append(grafted_row_shift['amalgamate_max'][0:1]))

    amalgamated_df.at[grafted_row.index, 'amalgamate_count'] = count
    amalgamated_df.at[grafted_row.index, 'amalgamate_total'] = total
    amalgamated_df.at[grafted_row.index, 'value'] = total / count
    amalgamated_df.at[grafted_row.index, 'amalgamate_min'] = min
    amalgamated_df.at[grafted_row.index, 'amalgamate_max'] = max

    return(amalgamated_df)

  def __amalgamate(self, grafted_df, threshold):
    amalgamated_df = grafted_df.copy()
    min_min_diff = threshold
    shifts = 0
    rows_combined = 1
    amalgamated_df['amalgamate_count'] = 1
    amalgamated_df['amalgamate_total'] = amalgamated_df['value']
    amalgamated_df['amalgamate_min'] = amalgamated_df['value']
    amalgamated_df['amalgamate_max'] = amalgamated_df['value']
    while rows_combined > 0:
      rows_combined = 0
      while min_min_diff <= threshold:
        shifts += 1
        [amalgamated_df, amalgamated_df_shift] = self.__initialise_shift(amalgamated_df, shifts)
        min_min_diff = np.min(amalgamated_df['min_diff'])
        matches = self.__find_matches(amalgamated_df, amalgamated_df_shift, shifts)
        match_count = np.sum(matches)
        row = 0
        while row < match_count:
          [c_lt, c_gt] = self.__find_close_shift_matched_box(amalgamated_df, amalgamated_df_shift, shifts, matches, row, threshold)
          if c_lt is not None:
            amalgamated_df = self.__combine_grafts(amalgamated_df, amalgamated_df_shift, shifts, matches, row, c_lt, c_gt)
            [amalgamated_df, amalgamated_df_shift] = self.__initialise_shift(amalgamated_df, shifts)
            matches = self.__find_matches(amalgamated_df, amalgamated_df_shift, shifts)
            match_count = np.sum(matches)
            rows_combined += 1
          else:
            row += 1
    return amalgamated_df
