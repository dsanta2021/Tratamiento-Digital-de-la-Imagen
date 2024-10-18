function filteredMovie = tempNoiseFilter(movie,numAvgFrames)


%
%   numAvgFrames:
%       how many frames to average in the temporal domain

frames = length(movie);
[r,c,temp] = size(movie(1).cdata);
filteredRGB = uint8(zeros(r,c,3,frames));

h = waitbar(0);
for k = 1:frames
    waitbar(k/frames,h,['Frame ',num2str(k)]);
    
   %perform filtering
    if k >= numAvgFrames
        for m = 1:numAvgFrames
            curimg_set(:,:,:,m) = movie(k-m+1).cdata;
        end
        
        filteredRGB(:,:,:,k) = mean(curimg_set,4);
    else
        filteredRGB(:,:,:,k) = movie(k).cdata;
    end
    
end
delete(h);
filteredMovie = immovie(filteredRGB);