<?php
$params = $_POST['labels'];
$out = fopen("labels.txt", "w");
fwrite($out, $params);
fclose($myfile);
echo "<h1>This is the response data<br></h1>"
?>
