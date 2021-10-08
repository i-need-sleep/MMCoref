# MMCoref

Submission for DSTC 10 simmc2 subtask 2: coreference resolution.

Links for checkpoints, etc are in the placeholder.txt files.

For each object, visual features are extracted using CLIP and/or BUTD. Non-visual prefab features are encoded using BERT or sentence BERT. 
These features are conbined with object index embeddings, positions, etc in a linear layer.
The comined features for each object and the flattened dialogue history are input into a pretrained UNITER model with a binary classification head for each object.

We train 5 models with different inputs (differenct visual backbone, etc) and ensemble them to produce the final result.
