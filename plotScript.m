% Loads the output produced by the program
source('output.txt');

if ( dim > 2 )
   disp ( 'Only 2D plots are available. Sorry!' );
   return;
end

% Total labels in the output
labels = max(dataset)(1);

% Produces the plots
figure; hold on;

for i = 0:labels
   which = (dataset(:,1) == i);
   plot ( dataset(which,2), dataset(which,3), 'o', 'markersize', 5 );
end

pause;
