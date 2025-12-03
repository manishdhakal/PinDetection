--- ✅ Grid Search Complete  for combined dataset ~ 21k ---
Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.4150

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4360
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.48      0.48      0.48       416
           1       0.40      0.39      0.40       296
           2       0.44      0.45      0.44       432
           3       0.45      0.45      0.45       403
           4       0.36      0.37      0.37       413
           5       0.34      0.39      0.36       572
           6       0.50      0.43      0.46       413
           7       0.42      0.39      0.40       408
           8       0.47      0.47      0.47       569
           9       0.57      0.52      0.54       413

    accuracy                           0.44      4335
   macro avg       0.44      0.43      0.44      4335
weighted avg       0.44      0.44      0.44      4335


--- ✅ Grid Search Complete ~ 21k (using pca) Training set size 17340 samples Testing set size 4335 samples PCA applied. Original features12 PCA features (retaining 95% variance) 9 ---
Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.3972

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4155
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.46      0.47      0.46       416
           1       0.40      0.41      0.40       296
           2       0.41      0.42      0.42       432
           3       0.42      0.42      0.42       403
           4       0.34      0.38      0.36       413
           5       0.33      0.35      0.34       572
           6       0.47      0.40      0.43       413
           7       0.39      0.36      0.37       408
           8       0.44      0.44      0.44       569
           9       0.55      0.52      0.54       413

    accuracy                           0.42      4335
   macro avg       0.42      0.42      0.42      4335
weighted avg       0.42      0.42      0.42      4335



--- ✅ Grid Search Complete for anjila-big [detector/data/test_dataset.csv] dataset ~12k  ---
Best parameters found on training data: {'C': 100, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.4733

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.5188
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.51      0.65      0.57       211
           1       0.44      0.48      0.46       157
           2       0.57      0.56      0.57       284
           3       0.64      0.62      0.63       256
           4       0.41      0.38      0.39       253
           5       0.50      0.44      0.47       329
           6       0.52      0.51      0.52       268
           7       0.46      0.49      0.47       208
           8       0.53      0.50      0.52       334
           9       0.57      0.59      0.58       231

    accuracy                           0.52      2531
   macro avg       0.51      0.52      0.52      2531
weighted avg       0.52      0.52      0.52      2531


--- ✅ Grid Search Complete anjila-big dataset detector/data/test_dataset.csv ~15k  ---
Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.4538

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4881
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.52      0.52      0.52       255
           1       0.39      0.39      0.39       203
           2       0.48      0.54      0.51       351
           3       0.55      0.50      0.52       308
           4       0.39      0.48      0.43       312
           5       0.43      0.44      0.44       413
           6       0.49      0.47      0.48       324
           7       0.53      0.47      0.50       257
           8       0.53      0.50      0.52       406
           9       0.61      0.54      0.57       283

    accuracy                           0.49      3112
   macro avg       0.49      0.49      0.49      3112
weighted avg       0.49      0.49      0.49      3112

--- ✅ Grid Search Complete anjila-big dataset detector/data/test_dataset.csv ~15k  ---
Training set size: 12445 samples, Testing set size: 3112 samples, PCA applied. Original features: 12, PCA features (retaining 95% variance): 9
--- ✅ Grid Search Complete ---
Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.4442

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4672
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.51      0.52      0.51       255
           1       0.38      0.39      0.38       203
           2       0.46      0.52      0.48       351
           3       0.53      0.47      0.50       308
           4       0.37      0.46      0.41       312
           5       0.43      0.44      0.43       413
           6       0.46      0.46      0.46       324
           7       0.52      0.44      0.47       257
           8       0.51      0.46      0.48       406
           9       0.58      0.51      0.54       283

    accuracy                           0.47      3112
   macro avg       0.47      0.47      0.47      3112
weighted avg       0.47      0.47      0.47      3112


--- ✅ Grid Search Complete anjila-individual ~10k ---
Best parameters found on training data: {'C': 100, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.4811

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4861
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.64      0.57       166
           1       0.37      0.46      0.41       113
           2       0.50      0.48      0.49       231
           3       0.64      0.56      0.60       212
           4       0.37      0.36      0.37       196
           5       0.41      0.33      0.37       264
           6       0.51      0.52      0.51       217
           7       0.48      0.52      0.50       162
           8       0.52      0.50      0.51       270
           9       0.51      0.55      0.53       187

    accuracy                           0.49      2018
   macro avg       0.48      0.49      0.48      2018
weighted avg       0.49      0.49      0.48      2018


--- ✅ Grid Search Complete anjila-individual ~10k - --- Training set size: 8072 samples Testing set size: 2018 samples PCA applied. Original features: 12 PCA features (retaining 95% variance): 9

Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.4751

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4911
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.54      0.60      0.57       166
           1       0.34      0.30      0.32       113
           2       0.47      0.51      0.49       231
           3       0.59      0.55      0.57       212
           4       0.38      0.42      0.40       196
           5       0.43      0.44      0.43       264
           6       0.50      0.51      0.50       217
           7       0.54      0.45      0.49       162
           8       0.52      0.55      0.54       270
           9       0.59      0.50      0.54       187

    accuracy                           0.49      2018
   macro avg       0.49      0.48      0.48      2018
weighted avg       0.49      0.49      0.49      2018

--- ✅ Grid Search Complete for manish dataset ~ 10k---
Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.3775

--- Final Model Evaluation (Best SVM) ---
Test Set Accuracy: 0.4031
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.50      0.57      0.53       234
           1       0.46      0.51      0.48       169
           2       0.28      0.31      0.29       183
           3       0.34      0.41      0.37       174
           4       0.36      0.34      0.35       195
           5       0.32      0.30      0.31       286
           6       0.27      0.28      0.27       176
           7       0.51      0.42      0.46       229
           8       0.40      0.32      0.35       277
           9       0.59      0.60      0.60       208

    accuracy                           0.40      2131
   macro avg       0.40      0.41      0.40      2131
weighted avg       0.40      0.40      0.40      2131


--- ✅ Grid Search Complete ---
Best parameters found on training data: {'C': 10, 'class_weight': 'balanced', 'gamma': 0.1, 'kernel': 'rbf'}
Best Cross-Validation F1-score: 0.3560

--- Final Model Evaluation (Best SVM) manish dataset ~ 10k (pca) ---
Test Set Accuracy: 0.3731
Test Set Classification Report:
              precision    recall  f1-score   support

           0       0.45      0.53      0.49       234
           1       0.40      0.45      0.42       169
           2       0.29      0.32      0.30       183
           3       0.32      0.38      0.35       174
           4       0.33      0.29      0.31       195
           5       0.29      0.27      0.28       286
           6       0.22      0.26      0.24       176
           7       0.46      0.39      0.42       229
           8       0.39      0.30      0.34       277
           9       0.56      0.58      0.57       208

    accuracy                           0.37      2131
   macro avg       0.37      0.38      0.37      2131
weighted avg       0.37      0.37      0.37      2131