function [  ] = PCR_add_reverberation( src, dest )

    %parameters = [0.3 0.7; 0.7 0.3];

    parameters = [];
    for i = 0.2:0.2:0.8
        for j = 0.2:0.2:0.8
            parameters = [parameters; i j];
        end
    end

    fileList = getAllFilesWithExtension(src, '.wav');
    for file_index = 1 : length(fileList)
        [path,filename,~] = fileparts(fileList{file_index});
        [signal_original,fs] = audioread(fileList{file_index});

        signal_original = signal_original(:,1);

        reverb = reverberator;
        reverb.SampleRate = fs;
        %To model a large room, use low decay factor, long reverb tail. 
        %To model a small room, use high decay factor, short reverb tail.

        %DecayFactor is proportional to the time it takes for 
        %reflections to run out of energy.

        %Diffusion is proportional to the rate at which the reverb tail 
        %builds in density. Increasing Diffusion pushes the reflections 
        %closer together, thickening the sound. Reducing Diffusion 
        %creates more discrete echoes.
        wetdry_candidates = [0.2 0.4 0.6 0.8];
        diffusion_candidates = [0.2 0.4 0.6];
        decay_candidates = [0.2 0.4 0.6];
        
        wetdry_pos = randi(length([0.2 0.4 0.6 0.8]), 1);
        diffusion_pos = randi(length([0.2 0.4 0.6]), 1);
        decay_pos = randi(length([0.2 0.4 0.6]), 1);
        
        WetDry = wetdry_candidates(wetdry_pos);
        Diffusion = diffusion_candidates(diffusion_pos);
        DecayFactor = decay_candidates(decay_pos);

        reverb.WetDryMix = WetDry;
        reverb.DecayFactor = DecayFactor;
        reverb.Diffusion = Diffusion;
        
        signal_reverb = step(reverb, signal_original);

        if(size(signal_original,2) == 1)
            signal_reverb = signal_reverb(:,1);
        end

        newfilename = strcat(dest,'\',filename,'_', 'WetDry', '_', num2str(WetDry*10), ...
            '_', 'DecayFactor', '_', num2str(DecayFactor*10),'_', 'Diffusion', '_', num2str(Diffusion*10),'.wav');
        disp(newfilename);
        %audiowrite(newfilename,signal_reverb,fs); 
    end
    
end

