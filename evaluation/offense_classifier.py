from parlai.core.agents import create_agent

opt = {'init_opt': None, 'show_advanced_args': False, 'task': 'interactive', 'download_path': '/mnt/home/liuhaoc1/ParlAI/downloads',
       'datatype': 'train', 'image_mode': 'raw', 'numthreads': 1, 'hide_labels': False, 'multitask_weights': [1], 'batchsize': 1,
       'datapath': '/mnt/home/liuhaoc1/ParlAI/data', 'model': None, 'model_file': '/mnt/home/liuhaoc1/ParlAI/data/models/dialogue_safety/single_turn/model',
       'init_model': None, 'dict_class': 'parlai.core.dict:DictionaryAgent', 'display_examples': False, 'display_prettify': False,
       'display_ignore_fields': 'label_candidates,text_candidates', 'interactive_task': True, 'local_human_candidates_file': None,
       'single_turn': True, 'image_size': 256, 'image_cropsize': 224, 'candidates': 'inline', 'eval_candidates': 'inline',
       'repeat_blocking_heuristic': True, 'fixed_candidates_path': None, 'fixed_candidate_vecs': 'reuse', 'encode_candidate_vecs': True,
       'train_predict': False, 'cap_num_predictions': 100, 'ignore_bad_candidates': False, 'rank_top_k': -1, 'embedding_size': 300, 'n_layers': 2,
       'ffn_size': 300, 'dropout': 0.0, 'attention_dropout': 0.0, 'relu_dropout': 0.0, 'n_heads': 2, 'learn_positional_embeddings': False,
       'embeddings_scale': True, 'n_positions': None, 'n_segments': 0, 'variant': 'aiayn', 'activation': 'relu', 'output_scaling': 1.0,
       'use_memories': False, 'wrap_memory_encoder': False, 'memory_attention': 'sqrt', 'normalize_sent_emb': False, 'share_encoders': True,
       'share_word_embeddings': True, 'learn_embeddings': True, 'reduction_type': 'first', 'interactive_mode': True, 'embedding_type': 'random',
       'embedding_projection': 'random', 'fp16': False, 'optimizer': 'adamax', 'learningrate': 0.0001, 'gradient_clip': 0.1, 'adam_eps': 1e-08,
       'momentum': 0, 'nesterov': True, 'nus': (0.7,), 'betas': (0.9, 0.999), 'weight_decay': None, 'lr_scheduler': 'reduceonplateau', 'lr_scheduler_patience': 3,
       'lr_scheduler_decay': 0.5, 'warmup_updates': -1, 'warmup_rate': 0.0001, 'update_freq': 1, 'rank_candidates': False, 'truncate': 1024, 'text_truncate': None,
       'label_truncate': None, 'history_size': -1, 'person_tokens': False, 'split_lines': False, 'use_reply': 'label', 'add_p1_after_newln': False,
       'delimiter': '\n', 'gpu': -1, 'no_cuda': False, 'dict_file': None, 'dict_initpath': None, 'dict_language': 'english', 'dict_max_ngram_size': -1,
       'dict_minfreq': 0, 'dict_maxtokens': -1, 'dict_nulltoken': '__null__', 'dict_starttoken': '__start__', 'dict_endtoken': '__end__',
       'dict_unktoken': '__unk__', 'dict_tokenizer': 're', 'dict_lower': False, 'bpe_debug': False, 'dict_textfields': 'text,labels', 'classes': None,
       'class_weights': None, 'ref_class': None, 'threshold': 0.5, 'print_scores': False, 'data_parallel': False,
       'get_all_metrics': True, 'load_from_pretrained_ranker': False, 'parlai_home': '/mnt/home/liuhaoc1/ParlAI',
       'override': {'reduction_type': 'first', 'model_file': '/mnt/home/liuhaoc1/ParlAI/data/models/dialogue_safety/single_turn/model', 'print_scores': False, 'single_turn': True},
       'starttime': 'Oct01_22-08'}

opt['interactive_mode'] = False
agent = create_agent(opt, requireModelExists=True)

def is_offense(texts):
    '''

    :param texts: list of text (str)
    :return:
    '''

    observe = [{'id': 'localHuman', 'episode_done': True, 'label_candidates': None, 'text': text} for text in texts]
    observations = []
    for ob in observe:
        agent.observe(ob)
        observations.append(agent.observation)

    results = agent.batch_act(observations)

    return [result['text'] for result in results]