import pandas as pd
import json
import pickle
from typing import Dict, Tuple
import logging
import os
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def create_mappings_from_question_csv(
    question_csv_path: str,
    assist_path: str,
    manual_skill_texts: Dict[int, str] = None,
    keyid2idx_path: str = None,
    encoding: str = "ISO-8859-1",
    save_dir: str = "./output"
) -> Tuple[Dict[str, str], Dict[str, int], pd.DataFrame]:
    """
    Create qid_to_kc and kc_name_to_id mappings based on a CSV file with question texts.
    """
    
    # Load question CSV
    logger.info(f"Loading questions from {question_csv_path}")
    question_df = pd.read_csv(question_csv_path)
    
    # Check what columns are available
    logger.info(f"Question CSV columns: {question_df.columns.tolist()}")
    
    # Find the ID column
    id_column = None
    for col in ['question_id', 'problem_id', 'problemId']:
        if col in question_df.columns:
            id_column = col
            break
    
    if not id_column:
        raise ValueError(f"No ID column found in question CSV. Available columns: {question_df.columns.tolist()}")
    
    # Standardize column names - rename to 'question_id'
    if id_column != 'question_id':
        question_df = question_df.rename(columns={id_column: 'question_id'})
    
    # Also check for problem_body column and rename if needed
    if 'problem_body' in question_df.columns and 'question_text' not in question_df.columns:
        question_df = question_df.rename(columns={'problem_body': 'question_text'})
    
    # Ensure question_id is numeric for merging
    if question_df['question_id'].dtype == 'object' and str(question_df['question_id'].iloc[0]).startswith('q'):
        # Remove 'q' prefix if present
        question_df['numeric_id'] = question_df['question_id'].str.replace('q', '').astype(int)
    else:
        # Convert to numeric and handle floats (425.0 -> 425)
        question_df['numeric_id'] = pd.to_numeric(question_df['question_id'], errors='coerce')
        # Convert float to int if they're whole numbers
        if question_df['numeric_id'].notna().any():
            question_df['numeric_id'] = question_df['numeric_id'].fillna(-1).astype(int)
            question_df.loc[question_df['numeric_id'] == -1, 'numeric_id'] = None
    
    # Load ASSIST dataset
    logger.info(f"Loading ASSIST data from {assist_path}")
    assist_df = pd.read_csv(assist_path, encoding=encoding, low_memory=False)
    
    # Check available columns and find the right ones
    available_columns = assist_df.columns.tolist()
    logger.info(f"ASSIST columns: {available_columns[:10]}...")  # Show first 10 columns
    
    # Find problem ID column
    problem_id_col = None
    for col in ['problem_id', 'problemId']:
        if col in available_columns:
            problem_id_col = col
            break
    
    if not problem_id_col:
        raise ValueError(f"No problem ID column found in ASSIST data")
    
    # Rename to standard names
    if problem_id_col != 'problem_id':
        assist_df = assist_df.rename(columns={problem_id_col: 'problem_id'})
    
    # Clean skill data
    assist_df['problem_id'] = pd.to_numeric(assist_df['problem_id'], errors='coerce')
    # Convert float problem IDs to int if they're whole numbers
    if assist_df['problem_id'].notna().any():
        # Check if all non-null values are whole numbers
        non_null_problems = assist_df['problem_id'].dropna()
        if (non_null_problems == non_null_problems.astype(int)).all():
            assist_df['problem_id'] = assist_df['problem_id'].fillna(-1).astype(int)
            assist_df.loc[assist_df['problem_id'] == -1, 'problem_id'] = None
    
    # Find and rename skill column
    skill_id_col = None
    for col in ['skill_id', 'skill', 'sequence_id']:
        if col in available_columns:
            skill_id_col = col
            break
    
    # Special handling for ASSIST 2017 where 'skill' IS the KC name (no skill_id)
    if skill_id_col == 'skill' and 'skill_id' not in available_columns:
        logger.info("ASSIST 2017 format detected: 'skill' column contains KC names directly")
        # Don't rename 'skill' column, keep it as is
        # Create a synthetic skill_id for internal processing
        unique_skills = assist_df['skill'].dropna().unique()
        skill_to_id = {skill: idx for idx, skill in enumerate(unique_skills, start=1)}
        assist_df['skill_id'] = assist_df['skill'].map(skill_to_id)
        assist_df['skill_name'] = assist_df['skill']  # skill column contains the names
        logger.info(f"Found {len(unique_skills)} unique skills in ASSIST 2017")
        logger.info(f"Sample skills: {list(unique_skills[:5])}")
    elif skill_id_col and skill_id_col != 'skill_id':
        assist_df = assist_df.rename(columns={skill_id_col: 'skill_id'})
        assist_df['skill_id'] = pd.to_numeric(assist_df['skill_id'], errors='coerce')
    elif skill_id_col is None:
        logger.warning("No skill column found!")
    
    # Get skill mappings for problems in question CSV
    valid_problem_ids = question_df['numeric_id'].dropna().unique()
    logger.info(f"Found {len(valid_problem_ids)} unique problems in question CSV")
    
    # Filter ASSIST to only these problems
    assist_filtered = assist_df[assist_df['problem_id'].isin(valid_problem_ids)]
    
    # For ASSIST 2017, ensure skill columns are properly set in filtered data too
    if 'skill' in available_columns and 'skill_id' not in available_columns:
        if 'skill_id' in assist_df.columns and 'skill_id' not in assist_filtered.columns:
            assist_filtered['skill_id'] = assist_filtered['skill'].map(skill_to_id)
            assist_filtered['skill_name'] = assist_filtered['skill']
    
    # Check which columns are available
    available_columns = assist_df.columns.tolist()
    logger.info(f"Available columns in ASSIST data: {available_columns[:10]}...")  # Show first 10 columns
    
    # Determine skill-related columns
    skill_id_col = None
    skill_name_col = None
    
    # Find skill ID column
    for col in ['skill_id', 'skill', 'sequence_id', 'skill_id']:
        if col in available_columns:
            skill_id_col = col
            break
    
    # Find skill name column (might not exist)
    for col in ['skill_name', 'skill_name', 'problem_name', 'skill']:
        if col in available_columns:
            skill_name_col = col
            break
    
    # Special handling for ASSIST 2017 where 'skill' column contains the skill names
    if 'skill' in available_columns and skill_name_col == 'skill':
        logger.info("ASSIST 2017 detected: 'skill' column contains skill names")
        # For ASSIST 2017, the 'skill' column has the actual skill names
        # We'll create a skill_id based on unique skill names
        assist_df['skill_name'] = assist_df['skill']
        assist_filtered['skill_name'] = assist_filtered['skill']
        
        # Create skill IDs for unique skill names
        unique_skills = assist_df['skill'].dropna().unique()
        skill_to_id = {skill: idx for idx, skill in enumerate(unique_skills, start=1)}
        assist_df['skill_id'] = assist_df['skill'].map(skill_to_id)
        assist_filtered['skill_id'] = assist_filtered['skill'].map(skill_to_id)
        
        skill_id_col = 'skill_id'
        skill_name_col = 'skill_name'
        
        # Debug: Show sample of skill mappings
        logger.info(f"Created {len(skill_to_id)} skill IDs from skill names")
        logger.info(f"Sample skill mappings: {dict(list(skill_to_id.items())[:5])}")
    
    if not skill_id_col:
        raise ValueError(f"No skill ID column found. Available columns: {available_columns}")
    
    logger.info(f"Using skill ID column: {skill_id_col}")
    logger.info(f"Using skill name column: {skill_name_col if skill_name_col else 'None - will use skill IDs'}")
    
    # Rename columns for consistency
    if skill_id_col != 'skill_id' and 'skill' not in available_columns:
        assist_df = assist_df.rename(columns={skill_id_col: 'skill_id'})
        assist_filtered = assist_filtered.rename(columns={skill_id_col: 'skill_id'})
    
    if skill_name_col and skill_name_col != 'skill_name' and 'skill' not in available_columns:
        assist_df = assist_df.rename(columns={skill_name_col: 'skill_name'})
        assist_filtered = assist_filtered.rename(columns={skill_name_col: 'skill_name'})
    elif not skill_name_col and 'skill' not in available_columns:
        # Create skill_name column with placeholder values
        assist_df['skill_name'] = None
        assist_filtered['skill_name'] = None
    
    # Get unique problem-skill mappings (keeping first skill if multiple)
    group_cols = ['skill_id', 'skill_name'] if 'skill_name' in assist_filtered.columns else ['skill_id']
    problem_skill_map = assist_filtered.groupby('problem_id').first()[group_cols].reset_index()
    
    # Debug: Check what we have in problem_skill_map
    logger.info(f"Problem-skill map columns: {problem_skill_map.columns.tolist()}")
    logger.info(f"Sample problem-skill mappings:")
    logger.info(problem_skill_map.head())
    
    # Create complete skill reference (all unique skills)
    skill_cols = ['skill_id', 'skill_name'] if 'skill_name' in assist_df.columns else ['skill_id']
    all_skills = assist_df[skill_cols].drop_duplicates()
    all_skills = all_skills.dropna(subset=['skill_id'])
    
    # Debug: Check skill data
    logger.info(f"Total unique skills: {len(all_skills)}")
    if 'skill_name' in all_skills.columns:
        logger.info(f"Sample skills with names:")
        logger.info(all_skills[all_skills['skill_name'].notna()].head())
    
    # Initialize mappings
    kc_name_to_id = {}
    kc_id_to_name = {}
    
    # Process all skills to create kc_name_to_id
    for _, row in all_skills.iterrows():
        if pd.notna(row.get('skill_id')):
            skill_id = int(row['skill_id'])
        else:
            continue
            
        skill_name = row.get('skill_name', None)
        
        # Determine skill name
        if pd.notna(skill_name):
            original_name = skill_name
        elif manual_skill_texts and skill_id in manual_skill_texts:
            original_name = manual_skill_texts[skill_id]
        else:
            original_name = f"Skill {skill_id}"
        
        # Clean name for use as key
        clean_name = clean_kc_name(original_name)
        
        # Handle duplicates
        if clean_name in kc_name_to_id and kc_name_to_id[clean_name] != skill_id:
            clean_name = f"{clean_name}_{skill_id}"
        
        kc_name_to_id[clean_name] = skill_id
        kc_id_to_name[skill_id] = clean_name
    
    logger.info(f"Created mappings for {len(kc_name_to_id)} unique KCs")
    
    # Merge question data with skill mappings
    merge_result = question_df.merge(
        problem_skill_map,
        left_on='numeric_id',
        right_on='problem_id',
        how='left'
    )
    
    # Ensure skill_name column exists (might be missing in some datasets)
    if 'skill_name' not in merge_result.columns:
        merge_result['skill_name'] = None
    
    question_df = merge_result
    
    # Create qid_to_kc mapping
    qid_to_kc = {}
    questions_with_kc = 0
    questions_without_kc = 0
    
    for _, row in question_df.iterrows():
        # Get question ID (with 'q' prefix if not already)
        if 'question_id' in row and pd.notna(row['question_id']):
            qid = str(row['question_id'])
            if not qid.startswith('q'):
                qid = f"q{qid}"
        else:
            qid = f"q{int(row['numeric_id'])}"
        
        # Get KC for this question
        if pd.notna(row.get('skill_id')):
            skill_id = int(row['skill_id'])
            if skill_id in kc_id_to_name:
                kc_name = kc_id_to_name[skill_id]
                qid_to_kc[qid] = kc_name
                questions_with_kc += 1
            else:
                questions_without_kc += 1
        else:
            questions_without_kc += 1
    
    logger.info(f"\nMapping results:")
    logger.info(f"- Questions with KC: {questions_with_kc}")
    logger.info(f"- Questions without KC: {questions_without_kc}")
    logger.info(f"- Total KCs: {len(kc_name_to_id)}")
    
    # Add KC name to dataframe
    question_df['kc_name'] = question_df['skill_id'].map(kc_id_to_name)
    
    # Debug: Check if KC names are being mapped
    logger.info(f"\nFinal question_df columns: {question_df.columns.tolist()}")
    logger.info(f"Questions with KC names: {question_df['kc_name'].notna().sum()}")
    logger.info(f"Sample of final data with KC names:")
    logger.info(question_df[['question_id', 'skill_id', 'kc_name', 'skill_name']].head(10))
    
    # Save outputs
    os.makedirs(save_dir, exist_ok=True)
    
    # Save qid_to_kc
    with open(os.path.join(save_dir, 'qid_to_kc.pkl'), 'wb') as f:
        pickle.dump(qid_to_kc, f)
    
    with open(os.path.join(save_dir, 'qid_to_kc.json'), 'w') as f:
        json.dump(qid_to_kc, f, indent=2)
    
    # Save kc_name_to_id
    with open(os.path.join(save_dir, 'kc_name_to_id.json'), 'w') as f:
        json.dump(kc_name_to_id, f, indent=2)
    
    # Save complete mapping info
    mapping_info = {
        'kc_name_to_id': kc_name_to_id,
        'kc_id_to_name': kc_id_to_name,
        'total_questions': len(question_df),
        'questions_with_kc': questions_with_kc,
        'questions_without_kc': questions_without_kc,
        'total_kcs': len(kc_name_to_id)
    }
    
    with open(os.path.join(save_dir, 'complete_mappings.json'), 'w') as f:
        json.dump(mapping_info, f, indent=2)
    
    # Save enhanced question dataframe
    question_df.to_csv(os.path.join(save_dir, 'questions_with_kc.csv'), index=False)
    
    logger.info(f"\nSaved all outputs to {save_dir}")
    
    return qid_to_kc, kc_name_to_id, question_df


def clean_kc_name(name: str) -> str:
    """Convert skill name to clean identifier format"""
    clean = str(name).lower()
    clean = ''.join(c if c.isalnum() or c == ' ' else ' ' for c in clean)
    clean = '_'.join(clean.split())
    while '__' in clean:
        clean = clean.replace('__', '_')
    return clean.strip('_')


def analyze_question_coverage(
    question_csv_path: str,
    assist_path: str,
    encoding: str = "ISO-8859-1"
) -> Dict:
    """
    Analyze how many questions have KC mappings and other statistics.
    """
    # Load data
    question_df = pd.read_csv(question_csv_path)
    assist_df = pd.read_csv(assist_path, encoding=encoding, low_memory=False)
    
    # Check available columns in question_df
    logger.info(f"Question CSV columns: {question_df.columns.tolist()}")
    
    # Find question/problem ID column
    question_id_col = None
    for col in ['question_id', 'problem_id', 'problemId']:
        if col in question_df.columns:
            question_id_col = col
            break
    
    if not question_id_col:
        raise ValueError(f"No question/problem ID column found in {question_csv_path}")
    
    # Get question IDs
    if question_df[question_id_col].dtype == 'object' and str(question_df[question_id_col].iloc[0]).startswith('q'):
        question_ids = pd.to_numeric(question_df[question_id_col].str.replace('q', ''), errors='coerce').dropna()
    else:
        question_ids = pd.to_numeric(question_df[question_id_col], errors='coerce').dropna()
    
    # Find problem ID column in ASSIST data
    problem_id_col = None
    for col in ['problem_id', 'problemId']:
        if col in assist_df.columns:
            problem_id_col = col
            break
    
    if not problem_id_col:
        raise ValueError(f"No problem ID column found in ASSIST data")
    
    assist_problems = set(pd.to_numeric(assist_df[problem_id_col], errors='coerce').dropna().unique())
    
    # Calculate coverage
    questions_in_assist = set(question_ids) & assist_problems
    questions_not_in_assist = set(question_ids) - assist_problems
    
    # Get skill coverage
    # Find skill ID column
    skill_id_col = None
    for col in ['skill_id', 'skill', 'sequence_id']:
        if col in assist_df.columns:
            skill_id_col = col
            break
    
    if skill_id_col:
        problems_with_skills = set(
            assist_df[assist_df[skill_id_col].notna()][problem_id_col].unique()
        )
    else:
        problems_with_skills = set()
    
    questions_with_skills = questions_in_assist & problems_with_skills
    
    stats = {
        'total_questions_in_csv': len(question_ids),
        'questions_in_assist': len(questions_in_assist),
        'questions_not_in_assist': len(questions_not_in_assist),
        'questions_with_skills': len(questions_with_skills),
        'coverage_percentage': (len(questions_with_skills) / len(question_ids) * 100) if len(question_ids) > 0 else 0
    }
    
    return stats


if __name__ == "__main__":
    
    # Run preprocess.py first
    print("Running preprocess.py...")
    #subprocess.run(['python', 'preprocess.py'])
    print("Preprocess.py completed!\n")
    
    # Define the lists
    assist_datas = [
        '/home/mahdi/Projects/pykt-toolkit-pt_emb/pretrained_Embedding/data_subsets/assist2009/skill_builder_data_corrected_collapsed.csv',
        '/home/mahdi/Projects/pykt-toolkit-pt_emb/pretrained_Embedding/data_subsets/assist2012/2012-2013-data-with-predictions-4-final.csv',
        '/home/mahdi/Projects/pykt-toolkit-pt_emb/pretrained_Embedding/data_subsets/assist2017/anonymized_full_release_competition_dataset.csv'
    ]

    question_csvs = [
        '/home/mahdi/Projects/pykt-toolkit-pt_emb/pretrained_Embedding/data_subsets/assist2009/questions.csv',
        '/home/mahdi/Projects/pykt-toolkit-pt_emb/pretrained_Embedding/data_subsets/assist2012/questions.csv',
        '/home/mahdi/Projects/pykt-toolkit-pt_emb/pretrained_Embedding/data_subsets/assist2017/questions.csv'
    ]
    
    # Output directories
    output_dirs = ['./mappings_output2009', './mappings_output2012', './mappings_output2017']
    
    # Manual skills
    manual_skills = {
        5: 'Number Line',
        102: 'Number Line',
        53: 'Ordering Integers',
        2: 'Circle Graph',
        37: 'Circle Graph',
        375: 'Number Line',
        10: 'Table',
        64: 'Table',
        9: 'Stem and Leaf Plot',
        14: 'Stem and Leaf Plot',
        74: 'Subtraction Whole Numbers',
        35: 'Effect of Changing Dimensions of a Shape Proportionally',
        69: 'Unit Conversion Within a System',
        173: 'Choose an Equation from Given Information',
        190: 'Choose an Equation from Given Information',
        193: 'Choose an Equation from Given Information',
        221: 'Choose an Equation from Given Information',
        321: 'Pythagorean Theorem',
    }
    
    # Process each dataset
    for i, (question_csv, assist_data, output_dir) in enumerate(zip(question_csvs, assist_datas, output_dirs)):
        dataset_name = output_dir.split('output')[-1]  # Gets 2009, 2012, or 2017
        print(f"\n{'='*60}")
        print(f"Processing ASSIST{dataset_name}")
        print(f"{'='*60}")
        
        # Analyze coverage
        stats = analyze_question_coverage(question_csv, assist_data)
        print(f"\nCoverage Analysis:")
        print(f"Total questions: {stats['total_questions_in_csv']}")
        print(f"Questions with KC mappings: {stats['questions_with_skills']} ({stats['coverage_percentage']:.1f}%)")
        
        # Create mappings
        qid_to_kc, kc_name_to_id, question_df = create_mappings_from_question_csv(
            question_csv_path=question_csv,
            assist_path=assist_data,
            manual_skill_texts=manual_skills,
            save_dir=output_dir
        )
        
        # Show sample results
        print(f"\nSample qid_to_kc mappings:")
        for qid, kc in list(qid_to_kc.items())[:5]:
            skill_id = kc_name_to_id[kc]
            print(f"  {qid} -> {kc} (skill_id: {skill_id})")
    
    print("\n✅ All datasets processed successfully!")