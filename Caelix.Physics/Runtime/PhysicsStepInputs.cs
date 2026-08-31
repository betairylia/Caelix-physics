using Unity.Collections;
using Caelix.Utils;

namespace Caelix
{
    public struct PhysicsStepInputs
    {
        public NativeHashMap<Guid128, VoxelEntityData> VoxelEntities;
        public NativeHashMap<Guid128, VoxelBodyData> VoxelBodies;
        public int nDynamicBodies;
    }
}
