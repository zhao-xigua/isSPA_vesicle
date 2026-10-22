import math, os, sys

try:
    from optparse import OptionParser
except:
    from optik import OptionParser
    
def main():
    (input_star_rln31,pixel_size,output_star) = parse_command_line()
    aa=open(input_star_rln31,"r")
    instar_line=aa.readlines()
    out=open(output_star,"w")
    mline=judge_mline0(instar_line)
    print(mline)
    CoordinateX_index=find_relion_metadata(instar_line,mline,"_rlnCoordinateX")
    CoordinateY_index=find_relion_metadata(instar_line,mline,"_rlnCoordinateY")
    OriginX_index=find_relion_metadata(instar_line,mline,"_rlnOriginXAngst")
    OriginY_index=find_relion_metadata(instar_line,mline,"_rlnOriginYAngst")
    for i in range(0,mline):
        out.write(instar_line[i])
    for i in range(mline,len(instar_line)):
        if (instar_line[i].split()):
            cols = instar_line[i].split()
            xx=float(cols[CoordinateX_index])
            yy=float(cols[CoordinateY_index])
            ox=float(cols[OriginX_index])
            oy=float(cols[OriginY_index])
            ox = ox / pixel_size
            oy = oy / pixel_size
            xx = xx - ox
            yy = yy - oy
            
            
            cols[CoordinateX_index] = f"{xx:.6f}"
            cols[CoordinateY_index] = f"{yy:.6f}"
            cols[OriginX_index] = "0.000000"
            cols[OriginY_index] = "0.000000"
            
            out.write("\t".join(cols) + "\n")
            
    aa.close()
    out.close()


def parse_command_line():
    usage="%prog <input_star_rln31> <pixel_size> <output_star> "
    parser = OptionParser(usage=usage, version="writen by zmj; used ZDJ's function")
    
    if len(sys.argv)<4: 
        print ("<input_star_rln31> <pixel_size> <output_star> ")
        print ("<input_star_rln31> for relion star with head")
        sys.exit(-1)
    
    (options, args)=parser.parse_args()
    input_star_rln31=str(args[0])
    pixel_size=float(args[1])
    output_star=str(args[2])
    return (input_star_rln31,pixel_size,output_star)

def find_relion_metadata(line,mline,tofind):
    for i in range(0,mline):
        if (line[i].split()):
            if (str(line[i].split()[0])==tofind):
                INDEX=int(line[i].split('#')[1])-1
                
    return(INDEX)
    
def judge_mline0(inline):
    trys=60
    intarget=-1
    n=1
    limit=20
    for i in range (0,trys):
        if(inline[i].split()):
            if(inline[i].split()[0][0]!="_"):
                if(intarget==1):
                    return i
                    break
                else:
                    continue
            if(inline[i].split()[0][0]=="_"):
                if (n < limit):
                    intarget=-1
                    n+=1
                if (n >= limit):
                    intarget=1

if __name__== "__main__":
    main()
