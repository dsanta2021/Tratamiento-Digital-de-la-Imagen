function noisyMov  = addnoise(movie,noiseType,noisyFrames)
%Addition of noise to a movie structure
%
%   noiseType:
%       'salt & pepper'
%       'gaussian'
%
%   noiseyFrames:
%       []              will add noise to all frames
%       [1 2 3 ...]     will add noise to frames 1, 2, 3, ...
%


frames = length(movie);
[r,c,temp] = size(movie(1).cdata);
noisy = uint8(zeros(r,c,3,frames));

h = waitbar(0);
for k = 1:frames
    waitbar(k/frames,h,['Frame ',num2str(k)]);
    
    curimg = movie(k).cdata;
    
    %if this is one of the selected frames, add noise
    if max(ismember(k,noisyFrames)) || isempty(noisyFrames)
        switch noiseType
            case 'salt & pepper'
                noisy(:,:,:,k) = imnoise(curimg,'salt & pepper',0.01);
            case 'gaussian'
                noisy(:,:,:,k) = imnoise(curimg,'gaussian',0,0.005);
        end
    end
    
end
delete(h);
noisyMov = immovie(noisy);