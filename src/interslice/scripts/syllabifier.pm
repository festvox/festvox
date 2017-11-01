package scripts::syllabifier;
require Exporter;
#use strict qw(vars);
#use vars qw(@ISA @EXPORT $VERSION $WIN32CONSOLE);
@ISA = qw(Exporter);
@EXPORT = qw(Phonification getSyllables getCurSyl isOnsetCodaVowel getPosinSyl isStart isMiddle isFinal getNphones syllableCount phoneCount get2Syllabs get2LastSyllabs getOnset getCoda getNSyllabs IsVowel getPhonePosition); 
#$VERSION = '1.03';


#open(PS,"Phoneset.txt") || die("File not available");
#open(VOW,"Vowels.txt") || die("File not found");
#open(CON,"Consonents.txt")|| die("File not found");

#@ps = <PS>;
#@vow = <VOW>;
#@con = <CON>;
#chomp(@ps);
#chomp(@vow);
#chomp(@con);

#$word = "bhaaii";
#print &getSyllables($word)."\n";
my $Phoneset = "tra t:ra a a: ax h: aa i ii u uu rx rx~ e ei ai o oo au k kh g gh ng~ ch chh j jh nj~ t th d dh n t: t:h d: d:h nd~ p ph b bh m y r l l: v sh shh s h qs~ r: n: n~ e~ o~ k~ kh~ g~ j~ d~ dh~ ph~ y~ lx lx~ nx l~ om:";
my $Vowels = "tra t:ra a ax h: aa i ii u uu rx rx~ e ei ai o oo au e~ o~ lx lx~ om:";
my $Consonants = "au k kh g gh ng~ ch chh j jh nj~ t th d dh n t: t:h d: d:h nd~ p ph b bh m y r l l: v sh shh s h qs~ r: n~ e~ k~ kh~ g~ j~ d~ dh~ ph~ y~ nx l~";

@ps = split(/\s+/, $Phoneset);
@vow = split(/\s+/, $Vowels);
@con = split(/\s+/, $Consonants);

sub getSyllables
{
  my $in = shift;
  $Phones = &Phonification($in,\@ps);
  $Syllables = &Syllabification(\@vow,\@con,$Phones);
  return $Syllables;
}
sub Phonification
{
  my $in = $_[0];
  my @phone = split(//,$in);
  #my @ps = @{$_[1]};;

  my %HashPhone;

  foreach (@ps)
  {
    $HashPhone{$_}++;
  }
  chomp(@ps);
  my($cur,$nex1,$nex2, $nex3);
  my($four,$tri,$di,$s);
  my $i;
  my $Phones;

  for($i=0;$i<@phone;$i++)
  {
    #&PhoneClassification($phone[$i],$phone[$i+1],$phone[$i+2],\@ps);

    $cur = $phone[$i];
    $nex1 = $phone[$i+1];
    $nex2 = $phone[$i+2];
    $nex3 = $phone[$i+3];

    $four = $cur.$nex1.$nex2.$nex3;
    $tri = $cur.$nex1.$nex2;
    $di = $cur.$nex1;
    $s = $cur;
    $j=0;
    
    if(exists $HashPhone{$four})
    {
        $Phones = $Phones.$four." ";
        $i = $i + 3;
    }
    elsif( exists $HashPhone{$tri})
    {
        $Phones = $Phones.$tri." ";
        $i = $i+2;
    }
    elsif(exists $HashPhone{$di})
    {
        $Phones = $Phones.$di." ";
        $i =$i+1;
    }
    elsif(exists $HashPhone{$s})
    {
        $Phones = $Phones.$s." ";
    }
    else
    {
	    return $in;
    }
  }
  return $Phones;
}
sub Syllabification
{
  my @vow = @{$_[0]};
  my @con = @{$_[1]};
  my $phone = $_[2];
  my @Phones = split(/ /,$phone);
  my $Syll;
  my $p;
  my %HashVow;

  foreach (@vow)
  {
    $HashVow{$_}++;
  }
  foreach (@ps)
  {
	  $HashPhone{$_}++;
  }

  for($p=0;$p<@Phones;$p++)
  {
          if(exists $HashVow{$Phones[$p]})
          {
            $Syll = $Syll.$Phones[$p]." ";
            next;
           }
           elsif($Phones[$p] eq "n:")
           {
            #chop($Syll);
            $Syll =~ s/ $//g;
            $Syll = $Syll.$Phones[$p]." ";
            next;
           }
           elsif($p == @Phones-1)
           {
            #chop($Syll);
            $Syll =~ s/ $//g;
            $Syll = $Syll.$Phones[$p];
           }
	   elsif(exists $HashPhone{$Phones[$p]})
           {
            $Syll = $Syll.$Phones[$p];
           }
	  else
	  {
		  return $phone;
	  }
  }
  
  my @words = split(/\s+/, $Syll);
  return $Syll if(@words == 0);
  if(&getVowelPosition($words[@words-1]) == -1 && @words >= 2)
  {
	$words[@words-2] = $words[@words-2].$words[@words-1];
	pop(@words);
  }
  
  $Syll = join(" ", @words);
  return $Syll;
}

sub getCurSyl
{
        my $phones = shift;
	my $pos = shift; #index in the phones

        my $sstr = &getsubstr($phones, $pos);
        @sylls = split(/\s+/, &getSyllables($sstr));
	return $sylls[@sylls];

}

sub getVowelFromSyllable
{
	my $syl = shift;
	my $vpos = &getVowelPosition($syl);
	my @phones = split(/\s+/, &Phonification($syl));
	return $phones[$vpos] if($vpos != -1);
	return -1;
}

sub isOnsetCodaVowel
{
	my $ph = shift;
	my $syl = shift;

        #my @phs = split(/\s+/, &Phonification($syl));
        #my $onset = "";
        #my $coda = "";
        #my $vpos = &getVowelPosition($syl);
	#print "Syl: $syl Pos: $vpos\n";

	#if($ph eq $phs[$vpos])
	#{
	#	return "coda";
	#}
	#return "onset";

	my $temp = &getOnset($syl);
	if($temp !~ /\$/)	
	{
		my @words = split(/#/, $temp);
		foreach my $word(@words)
		{
			if($word eq $ph)
			{
				return "onset";
			}
		}
	}
	$temp = &getVowel($syl);
	if($temp !~ /\$/ && $ph eq $temp)
	{
		return "vowel"; 	
	}
	my $temp = &getCoda($syl);
	#print $temp."\n";
	if($temp !~ /\$/)
	{
		my @words = split(/#/, $temp);
		foreach my $word(@words)
		{
			if($word eq $ph)
			{
				return "coda";
			}
		}
	}
	return "0";
}
sub getVowel
{
	return "\$" if($word eq "\$");
	my $word = shift;
	my @phs = split(/\s+/, &Phonification($word));
	for(my $i=0; $i<@phs; $i++)
	{
		return $phs[$i] if(&IsVowel($phs[$i]));
	}
	return "\$";
}
sub getOnset
{
	return "\$" if($word eq "\$");
	my $word = shift;
	my @phs = split(/\s+/, &Phonification($word));
	my $vow =0;
	my $onset = "";
	for(my $i=0; $i<@phs; $i++)
	{
		if(&IsVowel($phs[$i]))
		{
			$vow = 1;
			return "\$" if($onset eq "");
			$onset =~ s/#$//g;
			return $onset;
		}
		if($vow == 0)
		{			
			$onset = $onset.$phs[$i]."#";
		}
	}
	return "\$";
}

sub getCoda
{
	return "\$" if($word eq "\$");
	my $word = shift;
	my @phs = split(/\s+/, &Phonification($word));
	my $vow =0;
	my $coda = "";
	for(my $i=0; $i<@phs; $i++)
	{
		if(&IsVowel($phs[$i]))
		{
			$vow = 1;
			next;
		}
		if($vow == 1)
		{
			$coda = $coda.$phs[$i]."#";
		}
	}
	if($coda ne "")
	{
		$code =~ s/#$//g;
		return $coda;
	}
	else
	{
		return "\$";
	}	
}

sub getVowelPosition
{
        my $syl = shift;
        my @phs = split(/\s+/, &Phonification($syl));
        for($i=0; $i<@phs; $i++)
        {
               return $i if(&IsVowel($phs[$i]));
        }
	
	return -1;
}
sub getPosinSyl
{
	my $ph = shift;
	my $syl = shift;
	if($syl eq 0)
	{
		return 0;
	}
	$phones = &Phonification($syl, \@ps);
	@phs = split(/\s+/,$phones);
	for(my $i=0; $i<@phs; $i++)
	{
		return $i if($ph eq $phs[$i]);
	}
	return 0;	
}
sub getPhonePosition
{
	my $ph = shift;
	my $syl = shift;
	if($syl eq "0" || $syl eq "" || $syl !~ /$ph/)
	{
		return 0;
	}
	my @phones = split(/\s+/,&Phonification($syl));
	return 1 if($ph eq $phones[0]);
	return 3 if($ph eq $phones[@phones-1]);
	return 2 if(@phones > 2);
	return 0;
}
sub IsVowel
{
        my $ph = shift;
        for(my $i=0; $i<@vow; $i++)
        {
                return 1 if($vow[$i] eq $ph);
        }
        return 0;
}

sub isMiddle
{
        my $ph = shift;
	my $syl = shift;
        @phs = split(/\s+/, &Phonification($syl));	
	return 1 if(@phs >0 && $ph ne $phs[0] && $ph ne $phs[@phs-1]);	
        return 0;

}
sub isStart
{
        my $ph = shift;
	my $syl = shift;
        @phs = split(/\s+/, &Phonification($syl));	
	return 1 if($ph eq $phs[0]);	
        return 0;

}
sub isFinal
{
        my $ph = shift;
	my $syl = shift;
        @phs = split(/\s+/, &Phonification($syl));
        return 1 if(@phs > 0 && $ph eq $phs[@phs-1]);
        return 0;

}

sub getsubstr
{
	my $phones = shift;
	my $pos = shift;
	my @words = split(/\s+/, $phones);
	my $sstr = "";
	for($i=0; $i<=$pos; $i++)
	{
		$sstr = $sstr.$words[$i];
	}
	return $sstr;
}
sub getNphones
{
        my $syl = shift;
        my @phones = split(/\s+/,&Phonification($syl));
        return scalar(@phones);
}

sub finalSchwa
{
  my $word = shift;
  my @words = split(/ /, $word);
  my $lsyl = pop(@words);
  return $lsyl if($lsyl =~ /a[a-z:~]+$/);

  my $phones = &Phonification($lsyl);
  if($phones =~ / a / or $phones =~ / a$/)
  #if($phones =~ / a$/)
  {
        $lsyl =~ s/a//g;
        return $lsyl;
  }
  else
  {
        return $lsyl;
  }
}

sub syllablecount
{
  my $sylls = shift;
  #print $sylls;
  my @words = split(" ",$sylls);
  #print "Count: @words";
  return @words;
}
sub phoneCount
{
	my $word = shift;
	my @phs = split(/\s+/, &Phonification($word));
	return scalar(@phs);
}
sub getNSyllabs
{
  my $sylls = shift;
  my $n = shift;
  #print $sylls."\n";
  my @words = split(" ", $sylls);
  my $retData = "";
  #print "Hello".$words[0]." ".$words[1]."\n";
  if(@words >= $n)
  {
      chomp($words[$n-1]);
      #print "Return1: $words[0] $words[1]\n";
	#$words[$n-1] =~ s/,$//g;
      for(my $i=0; $i<$n; $i++)
      {
	      $retData = $retData.$words[$i]." ";
      }
      #return $words[0]." ".$words[1];
  }
  else #if(@words == 1)
  {
	  for(my $i=0; $i<@words; $i++)
	  {
		  $retData = $retData.$words[$i]." ";
	  }
	  for(my $i=@words; $i<$n; $i++)
	  {
		  $retData = $retData."\$"." ";
	  }
  }
  $retData =~ s/\s+/ /g;
  $retData =~ s/\s+$//g;
    return $retData;
}

sub get2Syllabs
{
  my $sylls = shift;
  #print $sylls."\n";
  my @words = split(" ", $sylls);
  #print "Hello".$words[0]." ".$words[1]."\n";
  if(@words >= 2)
  {
      chomp($words[1]);
      #print "Return1: $words[0] $words[1]\n";
	#$words[1] =~ s/,$//g;
      return $words[0]." ".$words[1];
  }
  elsif(@words == 1)
  {
    chomp($words[0]);
    #print "Return2: $words[0]\n";
    $words[0] =~ s/,$//g;
    return $words[0]." \$";
  }
  else
  {
    #print "Return3: $ $\n";
    return "\$ \$";
  }
}

sub get2LastSyllabs
{
  my $sylls = shift;
  chomp($sylls);
  my @words  = split(" ",$sylls);

  if(@words >=2)
  {
    $words[@words-2] =~ s/,$//g;
    return $words[@words-2]." ".$words[@words - 1];
  }
  elsif(@words == 1)
  {
	#$words[@words-1] =~ s/,$//g;
    return $words[@words-1]." \$";
  }
  else
  {
    return "\$ \$";
  }
}

sub get1Syllabs
{
  my $sylls = shift;
  #print $sylls."\n";
  my @words = split(" ", $sylls);
  #print "Hello".$words[0]." ".$words[1]."\n";
  if(@words >= 1)
  {
      chomp($words[1]);
      #print "Return1: $words[0] $words[1]\n";
      $words[0] =~ s/,$//g;
      return $words[0];
  }
  else
  {
    #print "Return3: $ $\n";
    return "\$";
  }
}

sub syllableCount
{
  my $sylls = shift;
  #print $sylls;
  my @words = split(" ",$sylls);
  #print "Count: @words";
  return @words;
}

sub revWord
{
  my $word = shift;
  my @words = split(" ", $word);
  return join(" ",reverse @words);
}
