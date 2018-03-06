<html>
   <head>
      <title>Your contribution</title>
      <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
      <link rel="stylesheet" type="text/css" href="css/style.css">
   </head>

   <?php
        include 'local_paths.php';
        $script = "source/php_called_scripts/get_frame.py";
        $args = "2>&1";
        $cmd = $python_interpreter." ".$script_base.$script." ".$args;
        $out = exec($cmd, $rets, $errorcode);
        if ($errorcode == 0){
            $error = "";
            $imgurl = $out;
        }else{
            $error = $out;
        }
        $nick = $_GET["nick"]
   ?>

   <body>
       <nav class="navbar navbar-expand-md navbar-dark fixed-top bg-dark">
         <a class="navbar-brand" href="#">HandTracking</a>
         <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarsExampleDefault" aria-controls="navbarsExampleDefault" aria-expanded="false" aria-label="Toggle navigation">
           <span class="navbar-toggler-icon"></span>
         </button>

         <div class="collapse navbar-collapse" id="navbarsExampleDefault">
           <ul class="navbar-nav mr-auto">
           </ul>
         </div>
       </nav>

       <div class="jumbotron">
         <div class="container">
           <h1>Let's begin</h1>
           <p>Here you're asked to arrange a green point (left click on the mouse) on every visible junction and a red one (right click on the mouse) on every non-visible junction on the LEFT picture.</p>
           <p>Please follow the order imposed by the yellow points appearing on the junctions on the right picture in arranging your points. Be as more accurate as you can.</p>
           <p>If you make a mistake use the 'Undo' button below; when you end either click on 'submit and get new sample' or 'submit and exit'.</p>
         </div>
       </div>
       <div class="container">
             <div class="row">
                 <div class="col md-auto">
                     <canvas id="left_c" width="500px" height="500px"></canvas>
                     <canvas id="left_c_points" width="500px" height="500px"></canvas>
                  </div>
                  <div class="col md-auto">
                      <canvas id="right_c" width="500px" height="500px"></canvas>
                      <canvas id="right_c_points" width="500px" height="500px"></canvas>
                  </div>
             </div>
             <div>
                 <!--<button id="undo_b">Undo</button>-->
                 <button id="undo_b" class="btn btn-primary btn-sm" role="button">Undo</button>
                 <!--<button id="new_sample_b">Submit and get other sample</button>-->
                 <button id="new_sample_b" class="btn btn-primary btn-sm" role="button">Submit and get other sample</button>
                 <!--<button id="submit_b">Submit and exit</button>-->
                 <button id="submit_b" class="btn btn-primary btn-sm" role="button">Submit and exit</button>
             </div>
       </div>

       <script src="javascript/helper_hand.js" type="text/javascript"></script>
       <script src="javascript/pinpointer.js" type="text/javascript"></script>
       <script type="text/javascript">
            var imgloaderror = "<?php echo $error;?>";
            if(imgloaderror != '')
                console.log("Loading error: "+imgloaderror);
            var nickname = "<?php echo $nick; ?>";
            var sample_img_url = "images/sample_hand.png";
            var target_img_url = "<?php echo $imgurl; ?>";
            var target_joints = [new Point(0.552764, 0.898438, false), new Point(0.334171, 0.822266, false),
                                 new Point(0.206030, 0.642578, false), new Point(0.125628, 0.537109, false),
                                 new Point(0.032663, 0.417969, false), new Point(0.399497, 0.421875, false),
                                 new Point(0.371859, 0.287109, false), new Point(0.361809, 0.167969, false),
                                 new Point(0.359296, 0.050781, false), new Point(0.557789, 0.404297, false),
                                 new Point(0.565327, 0.257812, false), new Point(0.562814, 0.140625, false),
                                 new Point(0.565327, 0.011719, false), new Point(0.713568, 0.433594, false),
                                 new Point(0.733668, 0.291016, false), new Point(0.771357, 0.177734, false),
                                 new Point(0.781407, 0.066406, false), new Point(0.846734, 0.472656, false),
                                 new Point(0.902010, 0.365234, false), new Point(0.932161, 0.283203, false),
                                 new Point(0.964824, 0.205078, false)];
       </script>
       <?php echo "Nickname: ".$nick; ?>
       <script src="javascript/main_setup.js" type="text/javascript"></script>
      <script src="javascript/on_click_events.js" type="text/javascript"></script>
      <script src="https://code.jquery.com/jquery-3.2.1.slim.min.js" integrity="sha384-KJ3o2DKtIkvYIK3UENzmM7KCkRr/rE9/Qpg6aAZGJwFDMVNA/GpGFF93hXpG5KkN" crossorigin="anonymous"></script>
      <script src="https://cdnjs.cloudflare.com/ajax/libs/popper.js/1.12.9/umd/popper.min.js" integrity="sha384-ApNbgh9B+Y1QKtv3Rn7W3mgPxhU9K/ScQsAP7hUibX39j7fakFPskvXusvfa0b4Q" crossorigin="anonymous"></script>
      <script src="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/js/bootstrap.min.js" integrity="sha384-JZR6Spejh4U02d8jOt6vLEHfe/JQGiRRSQQxSfFWpi1MquVdAyjUar5+76PVCmYl" crossorigin="anonymous"></script>
   </body>
</html>
