function run_eeglab_ica(input_set, output_set)
    % run_eeglab_ica
    % A MATLAB CLI script that boots EEGLAB silently to run ICA on raw brainwaves.
    %
    % Usage:
    %   run_eeglab_ica('F:\input_data.set', 'F:\cleaned_data.set')
    
    try
        fprintf('==================================================\n');
        fprintf('Initializing EEGLAB MATLAB Backend...\n');
        
        % 1. Boot up EEGLAB silently
        addpath('P:\auditory\eeglab2023.1');
        eeglab nogui;
        
        % 2. Parse paths and load dataset
        fprintf('Loading EEG dataset: %s\n', input_set);
        [in_dir, in_name, in_ext] = fileparts(input_set);
        EEG = pop_loadset('filename', [in_name, in_ext], 'filepath', in_dir);
        
        % 3. Run Independent Component Analysis (ICA)
        % Using the highly optimized 'runica' algorithm to calculate the artifact matrix
        fprintf('Running EEGLAB pop_runica() to mathematically isolate artifacts...\n');
        EEG = pop_runica(EEG, 'icatype', 'runica');
        
        % 4. Save the calculated weights back to a new .set file for Python
        [out_dir, out_name, out_ext] = fileparts(output_set);
        fprintf('Saving ICA-cleaned dataset: %s\n', output_set);
        EEG = pop_saveset(EEG, 'filename', [out_name, out_ext], 'filepath', out_dir);
        
        fprintf('EEGLAB processing complete! Returning control to Python.\n');
        fprintf('==================================================\n');
        
    catch ME
        fprintf('ERROR IN EEGLAB PROCESSING: %s\n', ME.message);
        exit(1);
    end
end
