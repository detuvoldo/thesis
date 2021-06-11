Modify using the code from: https://github.com/facebookresearch/UnsupervisedDecomposition

- Download HotpotQA dataset and save at /data folder.
> export MAIN_DIR=/path/to/UnsupervisedDecomposition
> cd $MAIN_DIR/XLM/

- Download and unzip BERT for QG Model, and save at QG/bert-base-uncased, from the link: https://drive.google.com/file/d/19a3b2Td8y_PUchTmG-nPTodrefP5DUWx/view?usp=sharing
- Download, unzip QG data training at QG/SQG_model folder, link: https://drive.google.com/file/d/1zbBWRCC6YHGGyT4s2MGt9DUPMhXOoBLU/view?usp=sharing

# Copy pretrained MLM model
mkdir -p $MAIN_DIR/XLM/dumped/mlm.dev1.pseudo_decomp_random.mined
wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/dumped/mlm.dev1.pseudo_decomp_random.mined/best-valid_mlm_ppl.pth
mv best-valid_mlm_ppl.pth dumped/mlm.dev1.pseudo_decomp_random.mined/
# Copy pre-trained decomposition model
mkdir -p dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223
for FILE in best-valid_mh-sh-mh_mt_effective_goods_back_bleu.pth hyp.st=0.0.bs=5.lp=1.0.es=False.seed=0.mh-sh.train.pred.bleu.sh.txt hyp.st=0.0.bs=5.lp=1.0.es=False.seed=0.mh-sh.valid.pred.bleu.sh.txt; do
    wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223/$FILE
    mv $FILE dumped/umt.dev1.pseudo_decomp.replace_entity_by_type/20639223/
done
# Copy single-hop question answering model ensemble
for NUM_PARAGRAPHS in 1 3; do
    mkdir -p checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS/
    for FILE in config.json pytorch_model.bin training_args.bin; do
        wget https://dl.fbaipublicfiles.com/UnsupervisedDecomposition/checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS/$FILE
        mv $FILE checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS/
    done
done

# Download FastText vectors for creating pseudo-decompositions
mkdir -p data/fastText
wget https://dl.fbaipublicfiles.com/fasttext/vectors-english/crawl-300d-2M.vec.zip
unzip crawl-300d-2M.vec.zip
mv crawl-300d-2M.vec data/fastText/
rm crawl-300d-2M.vec.zip
cd $MAIN_DIR


- Step 1: Install requirements package
            > pip install -r requirements.txt
- Step 2: Data pre-processing for SQG model
            > python data_processing.py --data_folder data --output_folder data
        
- Step 3: Generate pseudo-decompositions for multi-hop questions
        - Train SQG model:
            > cd QG/SQG_model
            > python SQG_train.py \
              --bert_model ../bert-base-uncased \
              --do_train \
              --train_file data/paragraph_81K_training_data.json \
              --output_dir SQG_model_paragraph_81K/ \
              --num_train_epochs 5 \
              --train_batch_size 128 \
              --max_seq_length 512 \
              --doc_stride 450 \
              --max_answer_length 16 \
              --max_query_length 42
              
        - Generate sub-questions given bridge entities and answers:
            > python SQG_gen_eval.py \
              --bert_model ../bert-base-uncased \
              --SQG_model SQG_model_paragraph_81K/pytorch_model.bin \
              --output_dir ../../out \
              --predict_folder ../../out \
              --max_seq_length 512 \
              --doc_stride 450 \
              --max_query_length 42 \
              --max_answer_length 16

- Step 4: Data pre-processing for decomposition model
        - Prepare multi-hop and single-hop question files:
            > python uqd_data_format.py --data_folder out --output_folder out/uqd/
            > mv out/uqd/ UnsupervisedDecomposition/XLM/data/umt
            > DATA_FOLDER=uqd
            > cd UnsupervisedDecomposition/XLM/
            > bash ./get-data-mt.sh --src mh --tgt sh --reload_codes dumped/xlm_en/codes_en --reload_vocab dumped/xlm_en/vocab_en --data_folder $DATA_FOLDER
            
            Read arguments:
            POSITIONAL=()
            while [[ $# -gt 0 ]]
            do
            key="$1"
            case $key in
             --main_dir)
               MAIN_DIR="$2"; shift 2;;
             *)
             POSITIONAL+=("$1")
             shift
             ;;
            esac
            done
            set -- "${POSITIONAL[@]}"
            
- Step 5: Train decomposition model
        - Installation:
            > conda create -y -n UnsupervisedDecomposition python=3.7
            > conda activate UnsupervisedDecomposition
            # Install PyTorch 1.0. We used CUDA 10.0 (with NCCL/2.4.7-1) (see https://pytorch.org/ to install with other CUDA versions):
            > conda install -y pytorch=1.0 torchvision cudatoolkit=10.0 -c pytorch
            > conda install faiss-gpu cudatoolkit=10.0 -c pytorch # For CUDA 10.0
            > pip install -r requirements.txt
            > python -m spacy download en_core_web_lg
            > cd $MAIN_DIR/XLM
            
        - Unsupervised Decomposition Training
            > export NPROC_PER_NODE=8  # Use 1 for single-GPU training
            > export N_NODES=1  # Use >1 for multi-node training (where each node has NPROC_PER_NODE GPUs)
            > BS=32
            
            # Select an MLM initialization checkpoint (for now, let's load the MLM we already pre-trained)
            > MLM_INIT=dumped/mlm.dev1.pseudo_decomp_random.mined/best-valid_mlm_ppl.pth 
            
            # Train USeq2Seq model
            > export NGPU=$NPROC_PER_NODE
            > if [[ $NPROC_PER_NODE -gt 1 ]]; then DIST_OPTS="-m torch.distributed.launch --nproc_per_node=$NPROC_PER_NODE"; else DIST_OPTS=""; fi
            > NUM_TRAIN=`wc -l < data/umt/$DATA_FOLDER/processed/train.mh`
            > python $DIST_OPTS train.py --exp_name umt.$DATA_FOLDER --data_path data/umt/$DATA_FOLDER/processed --dump_path ./dumped/ --reload_model "$MLM_INIT,$MLM_INIT" --encoder_only false --emb_dim 2048 --n_layers 6 --n_heads 16 --dropout 0.1 --attention_dropout 0.1 --gelu_activation true --use_lang_emb true --lgs 'mh-sh' --ae_steps 'mh,sh' --bt_steps 'mh-sh-mh,sh-mh-sh' --stopping_criterion 'valid_mh-sh-mh_mt_effective_goods_back_bleu,2' --validation_metrics 'valid_mh-sh-mh_mt_effective_goods_back_bleu' --eval_bleu true --epoch_size $((4*NUM_TRAIN/(NPROC_PER_NODE*N_NODES))) --lambda_ae '0:1,100000:0.1,300000:0' --optimizer 'adam_inverse_sqrt,beta1=0.9,beta2=0.98,lr=0.00003' --tokens_per_batch 1024 --batch_size $BS --word_shuffle 3 --word_dropout 0.1 --word_blank 0.1 --max_len 128 --bptt 128 --save_periodic 0 --split_data true --validation_weight 0.5
            
- Step 6: Train QA Model
            # Generate decompositions
            > ST=0.0
            > LP=1.0
            > BEAM=5
            > SEED=0
            
            # Point to model directory (change the final directory number/string/id below to match the directory string from the previous Unsupervised Seq2Seq training command)
            > MODEL_DIR=dumped/uqd/xxxxxxx
            > MODEL_NO="$(echo $MODEL_DIR | rev | cut -d/ -f1 | rev)"
            
            > for SPLIT in valid train; do
                # Note: Decrease batch size below if GPU goes out of memory
                > cat data/umt/uqd/processed/$SPLIT.mh | python translate.py --exp_name translate --src_lang mh --tgt_lang sh --model_path $MODEL_DIR/best-valid_mh-sh-mh_mt_effective_goods_back_bleu.pth --output_path $MODEL_DIR/$SPLIT.pred.bleu.sh --batch_size 48 --beam_size $BEAM --length_penalty $LP --sample_temperature $ST
            > done
            
            # Convert Sub-Qs to SQUAD format
            > cd $MAIN_DIR/pytorch-transformers
            > for SPLIT in valid train; do
                > python umt_gen_subqs_to_squad_format.py --model_dir $MODEL_DIR --data_folder uqd --sample_temperature $ST --beam $BEAM --length_penalty $LP --seed $SEED --split $SPLIT --new_data_format
            > done
            
            # Answer sub-Qs
            > DATA_FOLDER=data/hotpot.umt.uqd.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED
            > cp -R ../XLM/$DATA_FOLDER data/
            > for SPLIT in "dev" "train"; do
            > for NUM_PARAGRAPHS in 1 3; do
                # For fp16: Add "--fp16 --fp16_opt_level O2" below
                > python examples/run_squad.py --model_type roberta --model_name_or_path roberta-large --train_file $DATA_FOLDER/train.json --predict_file $DATA_FOLDER/$SPLIT.json --do_eval --do_lower_case --version_2_with_negative --output_dir checkpoint/roberta_large.hotpot_easy_and_squad.num_paragraphs=$NUM_PARAGRAPHS --per_gpu_train_batch_size 64 --per_gpu_eval_batch_size 32 --learning_rate 1.5e-5 --max_query_length 234 --max_seq_length 512 --doc_stride 50 --num_shards 1 --seed 0 --max_grad_norm inf --adam_epsilon 1e-6 --adam_beta_2 0.98 --weight_decay 0.01 --warmup_proportion 0.06 --num_train_epochs 2 --write_dir $DATA_FOLDER/roberta_predict.np=$NUM_PARAGRAPHS --no_answer_file
            > done
            > done
            
            
            # Ensemble sub-answer predictions
            for SPLIT in "dev" "train"; do
                python ensemble_answers_by_confidence_script.py --seeds_list 1 3 --no_answer_file --split $SPLIT --preds_file1 data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED/roberta_predict.np={}/nbest_predictions_$SPLIT.json
            done
            
            # Add sub-questions and sub-answers to QA input
            > FLAGS="--atype sentence-1-center --subq_model roberta-large-np=1-3 --use_q --use_suba --use_subq"
            > python add_umt_subqs_subas_to_q_squad_format_new.py --subqs_dir data/hotpot.umt.all.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED --splits train dev --num_shards 1 --model_dir $MODEL_DIR --sample_temperature $ST --beam $BEAM --length_penalty $LP --seed $SEED --subsample_data --use_easy --use_squad $FLAGS

            # Train QA model
            > export NGPU=1  # Set based on number of available GPUs
            > if [ $NGPU -gt 1 ]; then DIST_OPTS="-m torch.distributed.launch --nproc_per_node=$NGPU"; else DIST_OPTS=""; fi
            > if [ $NGPU -gt 1 ]; then EVAL_OPTS="--do_eval"; else EVAL_OPTS=""; fi
            > export MASTER_PORT=$(shuf -i 12001-19999 -n 1)
            > FLAGS_STRING="${FLAGS// --/.}"
            > FLAGS_STRING="${FLAGS_STRING//--/.}"
            > FLAGS_STRING="${FLAGS_STRING// /=}"
            > TN=hotpot.umt.uqd.model=$MODEL_NO.st=$ST.beam=$BEAM.lp=$LP.seed=$SEED$FLAGS_STRING.suba1=0.suba2=0-squad.medium_hard_frac=1.0
            > RANDOM_SEED=0
            > OUTPUT_DIR="checkpoint/tn=$TN/rs=$RANDOM_SEED"
            > cp -R ../XLM/data/$TN data/
            # For fp16: Add "--fp16 --fp16_opt_level O2" below
            > python $DIST_OPTS examples/run_squad.py --model_type roberta --model_name_or_path roberta-large --train_file data/$TN/train.json --predict_file data/$TN/dev.json --do_train $EVAL_OPTS --do_lower_case --version_2_with_negative --output_dir $OUTPUT_DIR --per_gpu_train_batch_size $((64/NGPU)) --per_gpu_eval_batch_size 32 --learning_rate 1.5e-5 --master_port $MASTER_PORT --max_query_length 234 --max_seq_length 512 --doc_stride 50 --num_shards 1 --seed $RANDOM_SEED --max_grad_norm inf --adam_epsilon 1e-6 --adam_beta_2 0.98 --weight_decay 0.01 --warmup_proportion 0.06 --num_train_epochs 2 --overwrite_output_dir
            
            