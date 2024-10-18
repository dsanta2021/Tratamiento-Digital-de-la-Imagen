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
    if k >= 2 && k < frames
        curimg_set(:,:,:,1) = movie(k-1).cdata;
        curimg_set(:,:,:,2) = movie(k).cdata;
        curimg_set(:,:,:,3) = movie(k+1).cdata;
        
        filteredRGB(:,:,:,k) = median(curimg_set,4);
    else
        filteredRGB(:,:,:,k) = movie(k).cdata;
    end
    
end
delete(h);
filteredMovie = immovie(filteredRGB);