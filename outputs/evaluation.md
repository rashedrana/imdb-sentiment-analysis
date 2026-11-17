# Evaluation — IMDb Sentiment Analysis

Model: TF-IDF (unigrams + bigrams, 20,000 features) + LogisticRegression,
trained on the full 25,000-review training split (Stage 2) and evaluated
here on the held-out 25,000-review test split.

## Headline metrics

| Metric | Value |
| --- | --- |
| Accuracy | 0.8992 |
| ROC AUC | 0.9629 |
| Test set size | 25,000 reviews |

Chance level is 50% (the test set is balanced 50/50).

## Per-class results

```
              precision    recall  f1-score   support

    negative     0.9031    0.8942    0.8987     12500
    positive     0.8953    0.9041    0.8997     12500

    accuracy                         0.8992     25000
   macro avg     0.8992    0.8992    0.8992     25000
weighted avg     0.8992    0.8992    0.8992     25000
```

## Confusion matrix

Rows = true label, columns = predicted label.

|  | Predicted negative | Predicted positive |
| --- | --- | --- |
| True negative | 11,178 | 1,322 |
| True positive | 1,199 | 11,301 |

## Error analysis

2,521 of 25,000 test reviews were misclassified
(10.1%). Five examples:

### 1. True: negative | Predicted: positive | P(positive) = 0.826

Confident error — model was 83% sure of the wrong label; likely sarcasm, irony, or praise of one aspect (e.g. acting) while the overall verdict is the opposite.

> "First off let me say, If you haven't enjoyed a Van Damme movie since bloodsport, you probably will not like this movie. Most of these movies may not have the best plots or best actors but I enjoy these kinds of movies for what they are. This movie is much better than any of the movies the other acti..."

### 2. True: negative | Predicted: positive | P(positive) = 0.640

Borderline call — model only 64% confident in the wrong label; review reads near-neutral.

> "Ben, (Rupert Grint), is a deeply unhappy adolescent, the son of his unhappily married parents. His father, (Nicholas Farrell), is a vicar and his mother, (Laura Linney), is ... well, let's just say she's a somewhat hypocritical soldier in Jesus' army. It's only when he takes a summer job as an assis..."

### 3. True: negative | Predicted: positive | P(positive) = 0.569

Mixed sentiment (mentions both praise and criticism: 1 positive cue(s), 1 negative cue(s)); model was 57% confident in the wrong label.

> "The Forgotten (AKA: Don't Look In The Basement) is a very cheaply made and very old looking horror movie.<br /><br />The story is very slow and never really reaches anything worth getting excited about.<br /><br />The patients at the asylum are embarrassingly funny especially Sam and the old woman w..."

### 4. True: negative | Predicted: positive | P(positive) = 0.638

Borderline call — model only 64% confident in the wrong label; review reads near-neutral.

> "I'm the type of guy who loves hood movies from New Jack City to Baby Boy to Killa Season, from the b grade to the Hollywood. but this movie was something different. i am no hater and this movie was kinda enjoyable. but some bits were just weird. well the acting wasn't to good, compared to Silkk The ..."

### 5. True: negative | Predicted: positive | P(positive) = 0.628

Borderline call — model only 63% confident in the wrong label; review reads near-neutral.

> "This film features two of my favorite guilty pleasures. Sure, the effects are laughable, the story confused, but just watching Hasselhoff in his Knight Rider days is always fun. I especially like the old hotel they used to shoot this in, it added to what little suspense was mustered. Give it a 3...."

