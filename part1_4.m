im = imread('charact2.bmp');
imshow(im);
[h,w,c] = size(im);
h1 = ones(3,3)/9;
im1r = filter2(h1, im(:,:,1));
im1g = filter2(h1, im(:,:,2));
im1b = filter2(h1, im(:,:,3));
im1 = cat(3, im1r, im1g, im1b);
im11 = average_mask(double(im), double(3));
figure(2)
imshow(uint8(im11));
im2 = rotating_mask(double(im));
figure(3)
imshow(uint8(im2));
im3=im(h/2:h,:,:);
%imshow(im3)
im4=zeros(h/2,w);
for i=1:184
    for j=1:990
        im4(i,j)=0.2989*im(i,j,1)+0.587*im(i,j,2)+0.114*im(i,j,3);
    end
end
im4 = im4>128;
%imshow(im4);

function im2 = average_mask(im1, r)
    [h,w,c] = size(im1);
    im2 = zeros(h,w,c);
    mask = double(1.0/(r * r));
    edge = int16(r/2);
    width = edge - 1;
    for x = 1:c    
        for i = edge:h-edge
            for j = edge:w-edge
                pix = 0.0;
                for a = i - width:i + width
                    for b = j - width:j + width
                        pix = pix + im1(a, b, x) * mask;
                    end
                end
                im2(i, j, x) = uint8(pix);
            end
        end
    end         
end



function llist = location_list(x, y)
    llist = [[x, x+2, y, y+2]; [x-1, x+1, y, y+2]; [x-2, x, y, y+2]; ...
        [x-2, x, y-1, y+1]; [x-2, x, y-2, y]; [x-1, x+1, y-2, y]; ...
        [x, x+2, y-2, y]; [x, x+2, y-1, y+1]; [x-1, x+1, y-1, y+1]];
end

function [value, b] = dispersion_value(im, r)
    %r
    a = 0.0;
    b = 0.0;
    [i1, i2, j1, j2] = deal(r(1), r(2), r(3), r(4));
    for i = i1:i2
        for j = j1:j2
            a = a + im(i, j)*im(i, j);
            b = b + im(i, j);
        end
    end
    %b*b
    %(a-(b*b/9))
    value = (a-(b*b/9)) / 9;
end


function im2 = rotating_mask(im1)
    [h,w,c] = size(im1);
    im2 = zeros(h,w,c);
    for x = 1:c    
        for i = 3:h-3
            for j = 3:w-3
                dispersion_values = [];
                llist = location_list(i, j);
                for a = 1:size(llist)
                    %a
                    %llist
                    [t1, t2] = dispersion_value(im1(:,:,x), llist(a,:));
                    dispersion_values = [dispersion_values; [t1, t2]];
                end
                dispersion_values = sortrows(dispersion_values, 1);
                %dispersion_values
                im2(i, j, x) = uint8(dispersion_values(1, 2)/9);
            end
        end
    end
end

